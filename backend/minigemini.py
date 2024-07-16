import re
from accelerate import infer_auto_device_map

import transformers
import warnings
# disable some warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')

from mgm.constants import IMAGE_TOKEN_INDEX
from mgm.model.builder import load_pretrained_model
from mgm.mm_utils import process_images, tokenizer_image_token

from vision_qna import *

# YanweiLi/MGM-2B (no 4bit)
#  out = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)
# AttributeError: 'Parameter' object has no attribute 'quant_state'

# YanweiLi/MGM-7B
# YanweiLi/MGM-7B-HD
# YanweiLi/MGM-13B
# YanweiLi/MGM-34B
# YanweiLi/MGM-34B-HD
# YanweiLi/MGM-13B-HDs
# YanweiLi/MGM-8x7B-HD
# YanweiLi/MGM-8x7B

# TODO:
# YanweiLi/MGM-8B-HD
# YanweiLi/MGM-8B

class VisionQnA(VisionQnABase):
    model_name: str = "minigemini"
    format: str = "llama2"
    vision_layers: List[str] = ["vision_tower", "vision_tower_aux", "vlm_uni_aux_projector", "vlm_uni_val_projector"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        model_base, model_name = model_id.split('/', 1)
        del self.params['low_cpu_mem_usage']
        del self.params['pretrained_model_name_or_path']
        del self.params['trust_remote_code']
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_id, None, model_name, **self.params)
    
        self.loaded_banner()
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = '<image>\n' + prompt

        if hasattr(self.model.config, 'image_size_aux'):
            if not hasattr(self.image_processor, 'image_size_raw'):
                self.image_processor.image_size_raw = self.image_processor.crop_size.copy()
            self.image_processor.crop_size['height'] = self.model.config.image_size_aux
            self.image_processor.crop_size['width'] = self.model.config.image_size_aux
            self.image_processor.size['shortest_edge'] = self.model.config.image_size_aux

        image_tensor = process_images(images, self.image_processor, self.model.config)
        
        image_grid = getattr(self.model.config, 'image_grid', 1)
        if hasattr(self.model.config, 'image_size_aux'):
            raw_shape = [self.image_processor.image_size_raw['height'] * image_grid,
                        self.image_processor.image_size_raw['width'] * image_grid]
            image_tensor_aux = image_tensor 
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                        size=raw_shape,
                                                        mode='bilinear',
                                                        align_corners=False)
        else:
            image_tensor_aux = []

        if image_grid >= 2:            
            raw_image = image_tensor.reshape(3, 
                                            image_grid,
                                            self.image_processor.image_size_raw['height'],
                                            image_grid,
                                            self.image_processor.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(-1, 3,
                                        self.image_processor.image_size_raw['height'],
                                        self.image_processor.image_size_raw['width'])
                    
            if getattr(self.model.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(global_image, 
                                                            size=[self.image_processor.image_size_raw['height'],
                                                                    self.image_processor.image_size_raw['width']], 
                                                            mode='bilinear', 
                                                            align_corners=False)
                # [image_crops, image_global]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()
            image_tensor = image_tensor.unsqueeze(0)
    
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            image_tensor_aux = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor_aux]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            image_tensor_aux = image_tensor_aux.to(self.model.device, dtype=torch.float16)


        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        params = self.get_generation_params(request)
        
        if 'top_k' in params: del params['top_k'] # avoid warnings
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                images_aux=image_tensor_aux if len(image_tensor_aux)>0 else None,
                bos_token_id=self.tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=self.tokenizer.pad_token_id,  # Pad token
                **params,
            )
        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return answer



