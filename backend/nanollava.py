from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

import transformers
import warnings
# disable some warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# 'qnguyen3/nanoLLaVA'

def join_int_lists(int_lists, separator):
    result = []
    for i, lst in enumerate(int_lists):
        result.extend(lst)
        if i < len(int_lists) - 1:
            result.append(separator)
    return result

class VisionQnA(VisionQnABase):
    model_name: str = "nanollava"
    format: str = "chatml"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)
        
        torch.set_default_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
            self.model = self.model.to(self.device)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        default_params = {
            'top_p': 0.8,
            'temperature': 0.7,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        encoded_images = self.model.process_images(images, self.model.config).to(dtype=self.model.dtype)

        params = self.get_generation_params(request, default_params=default_params)

        text_chunks = [self.tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        text_with_img_tok = join_int_lists(text_chunks, -200) # -200 == <image>
        input_ids = torch.tensor(text_with_img_tok, dtype=torch.long).unsqueeze(0)
        output = self.model.generate(input_ids, images=encoded_images, **params)

        response = self.tokenizer.decode(output[0][input_ids.size(1):].cpu(), skip_special_tokens=True)
        return response
