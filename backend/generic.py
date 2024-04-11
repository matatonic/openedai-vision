from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

class VisionQnA(VisionQnABase):
    model_name: str = "generic"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        encoded_images = self.model.encode_image(images)
        inputs = self.tokenizer(prompt, encoded_images, return_tensors="pt")

        params = self.get_generation_params(request)

        output = self.model.generate(**inputs, **params)
        response = self.tokenizer.decode(output[0][inputs.input_ids.size(1):].cpu(), skip_special_tokens=True)

        return response
