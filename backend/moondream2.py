import re
from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

# vikhyatk/moondream2

class VisionQnA(VisionQnABase):
    model_name: str = "moondream2"
    revision: str = '2024-03-13' # 'main'
    format: str = 'phi15'
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        # not supported yet
        del self.params['device_map']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
    
#        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, format=self.format)

        encoded_images = self.model.encode_image(images).to(self.device)

        params = self.get_generation_params(request)
        
        answer = self.model.generate(
            encoded_images,
            prompt,
            eos_text="<END>",
            tokenizer=self.tokenizer,
            **params,
        )[0]
        answer = re.sub("<$|<END$", "", answer).strip()
        return answer
