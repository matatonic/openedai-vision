from transformers import AutoProcessor, AutoModelForCausalLM

from vision_qna import *

# microsoft/Phi-3-vision-128k-instruct

class VisionQnA(VisionQnABase):
    model_name: str = "phi3"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await phi3_prompt_from_messages(request.messages)

        inputs = self.processor(prompt, images=images, return_tensors="pt").to(self.model.device)

        default_params = { 
            "temperature": 0.0, 
            "do_sample": False, 
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        } 

        params = self.get_generation_params(request, default_params)

        output = self.model.generate(**inputs, **params)
        response = self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return response
