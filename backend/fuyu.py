from transformers import FuyuProcessor, FuyuForCausalLM

from vision_qna import *

# "adept/fuyu-8b"

class VisionQnA(VisionQnABase):
    model_name: str = "fuyu"
    format: str = "fuyu"
    vision_layers: List[str] = ["vision_embed_tokens"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)
        
        del self.params['trust_remote_code'] # not needed.

        self.processor = FuyuProcessor.from_pretrained(model_id)
        self.model = FuyuForCausalLM.from_pretrained(**self.params)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        inputs = self.processor(text=prompt, images=images[0], return_tensors="pt").to(self.model.device)

        params = self.get_generation_params(request)

        output = self.model.generate(**inputs, **params)
        response = self.processor.decode(output[0][inputs.input_ids.size(1):].cpu(), skip_special_tokens=True)

        return response.strip()
