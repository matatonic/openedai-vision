from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

# Assumes mistral prompt format!!
# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

from vision_qna import VisionQnABase

class VisionQnA(VisionQnABase):
    model_name: str = "llava"
    
    def __init__(self, model_id: str, device: str, extra_params = {}):
        self.device = self.select_device() if device == 'auto' else device

        params = {
            'pretrained_model_name_or_path': model_id,
            'torch_dtype': torch.float32 if device == 'cpu' else torch.float16,
            'low_cpu_mem_usage': True,
        }
        if extra_params.get('load_in_4bit', False):
            load_in_4bit_params = {
                'bnb_4bit_compute_dtype': torch.float32 if device == 'cpu' else torch.float16,
                'load_in_4bit': True,
            }
            params.update(load_in_4bit_params)

        if extra_params.get('load_in_8bit', False):
            load_in_8bit_params = {
                'load_in_8bit': True,
            }
            params.update(load_in_8bit_params)

#            'use_flash_attention_2': True,
        if extra_params.get('use_flash_attn', False):
            flash_attn_params = {
                "attn_implementation": "flash_attention_2",
            }
            params.update(flash_attn_params)

        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(**params)
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
            self.model.to(self.device)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")

    async def single_question(self, image_url: str, prompt: str) -> str:
        image = await self.url_to_image(image_url)
        
        # prepare image and text prompt, using the appropriate prompt template
        prompt = f"[INST] <image>\n{prompt} [/INST]"
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)

        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=300)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        id = answer.rfind('[/INST]')
        return answer[id + 8:]
