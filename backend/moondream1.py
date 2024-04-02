import re
from transformers import CodeGenTokenizerFast, AutoModelForCausalLM

from vision_qna import *

class VisionQnA(VisionQnABase):
    model_name: str = "moondream1"
    format: str = 'phi15'
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)

        # not supported yet
        del self.params['device_map']

        self.tokenizer = CodeGenTokenizerFast.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(**self.params, trust_remote_code=True)
        
        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model.to(self.device)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")

    async def chat_with_images(self, messages: list[Message], max_tokens: int) -> str:
        images, prompt = await prompt_from_messages(messages, self.format)
        encoded_images = self.model.encode_image(images[0]).to(self.device)

        # XXX currently broken here... 
        """
          File "hf_home/modules/transformers_modules/vikhyatk/moondream1/f6e9da68e8f1b78b8f3ee10905d56826db7a5802/modeling_phi.py", line 318, in forward
    padding_mask.masked_fill_(key_padding_mask, 0.0)
RuntimeError: The expanded size of the tensor (747) must match the existing size (748) at non-singleton dimension 1.  Target sizes: [1, 747].  Tensor sizes: [1, 748]
        """
        answer = self.model.generate(
            encoded_images,
            prompt,
            eos_text="<END>",
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens,
        )[0]
        answer = re.sub("<$|<END$", "", answer).strip()
        return answer

