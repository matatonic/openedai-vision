import re
from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

# vikhyatk/moondream2

class VisionQnA(VisionQnABase):
    model_name: str = "moondream2"
    revision: str = '2024-03-13' # 'main'
    format: str = 'phi15'
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)

        # not supported yet
        del self.params['device_map']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
    
#        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, messages: list[Message], max_tokens: int) -> str:
        images, prompt = await phi15_prompt_from_messages(messages)

        encoded_images = self.model.encode_image(images).to(self.device)

        answer = self.model.generate(
            encoded_images,
            prompt,
            eos_text="<END>",
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens,
            #**kwargs,
        )[0]
        answer = re.sub("<$|<END$", "", answer).strip()
        return answer
