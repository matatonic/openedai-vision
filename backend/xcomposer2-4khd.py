import os
from math import ceil
import warnings
import torch
from transformers import AutoTokenizer, AutoModel, logging

from vision_qna import *

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# internlm/internlm-xcomposer2-4khd-7b
MAX_TILES = 55

# --4bit:
# Linear4bit.forward() takes 2 positional arguments but 3 were given

def calc_hd(image, max_num=MAX_TILES):
    # Not sure if this is correct, but there are no instructions for how to set it
    img = Image.open(image)
    width, height = img.size
    del img

    return min(ceil(width // 336) * ceil(height // 336), max_num)

class VisionQnA(VisionQnABase):
    model_name: str = "internlm-xcomposer2-4khd-7b"
    format: str = "chatml"
    vision_layers: List[str] = ['vit', 'vision_proj']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)
        
        torch.set_default_dtype(self.dtype)

        self.max_tiles = extra_params.get('max_tiles', MAX_TILES)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
            self.model = self.model.to(self.device)

        self.eos_token = '[UNUSED_TOKEN_145]'
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

        self.loaded_banner()
    

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        prompt, history, files, meta_instruction = await prompt_history_images_system_from_messages(request.messages, img_tok='<ImageHere>', url_handler=url_to_file)

        meta_instruction = meta_instruction if meta_instruction else ('You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
        '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.'),

        if files:
            image = self.model.encode_img(files[-1], hd_num=calc_hd(files[-1], max_num=self.max_tiles))
            inputs, im_mask = self.model.interleav_wrap_chat(self.tokenizer, prompt, image, history, meta_instruction)
        else:
            inputs = self.model.build_inputs(self.tokenizer, prompt, history, meta_instruction)
            im_mask = torch.zeros(inputs['input_ids'].shape[:2]).cuda().bool()
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items() if torch.is_tensor(v)
        }
        inputs['im_mask'] = im_mask

        default_params = {
            #'num_beams': 3,
            #'do_sample': False,
            "temperature": 1.0,
            "top_p": 0.8,
            'do_sample': True,
            'repetition_penalty': 1.005,
            'eos_token_id': [ self.tokenizer.eos_token_id, self.eos_token_id ], # also add end-of-assistant token in eos token id to avoid unnecessary generation
        }
        params = self.get_generation_params(request, default_params)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        try:
            def wrapper(**kwargs):
                with torch.cuda.amp.autocast():
                    _ = self.model.generate(**kwargs)

            for new_text in threaded_streaming_generator(generate=wrapper, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
                end = new_text.find(self.eos_token)
                if end == -1:
                    yield new_text
                else:
                    yield new_text[:end]
                    break

        except Exception as e:
            logger.error(e)
            # raise

        finally:
            for f in files:
                os.remove(f)

