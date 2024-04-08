from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from vision_qna import *

# model_id = "llava-hf/llava-v1.6-34b-hf" # chatml
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf" # vicuna
# model_id = "llava-hf/llava-v1.6-vicuna-7b-hf" #  vicuna
# model_id = "llava-hf/llava-v1.6-mistral-7b-hf" # llama2

class VisionQnA(VisionQnABase):
    model_name: str = "llavanext"
    format: str = 'llama2'
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        del self.params['trust_remote_code']

        use_fast = 'mistral' in model_id
        self.processor = LlavaNextProcessor.from_pretrained(model_id, use_fast=use_fast)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(**self.params).eval()

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")

    async def chat_with_images(self, request: ImageChatRequest) -> str:

        images, prompt = await prompt_from_messages(request.messages, self.format)
        inputs = self.processor(prompt, images, return_tensors="pt").to(self.model.device)

        params = self.get_generation_params(request)

        output = self.model.generate(**inputs, **params)
        response = self.processor.decode(output[0][inputs['input_ids'].size(1):].cpu(), skip_special_tokens=True)
        
        return response
