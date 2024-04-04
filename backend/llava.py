from transformers import LlavaProcessor, LlavaForConditionalGeneration
from vision_qna import *

# llava-hf/bakLlava-v1-hf # llama2
# llava-hf/llava-1.5-7b-hf # vicuna
# llava-hf/llava-1.5-13b-hf # vicuna

class VisionQnA(VisionQnABase):
    model_name: str = "llava"
    format: str = 'vicuna'
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)

        if not format:
            # guess the format based on model id
            if 'mistral' in model_id.lower():
                self.format = 'llama2'
            elif 'bakllava' in model_id.lower():
                self.format = 'llama2'
            elif 'vicuna' in model_id.lower():
                self.format = 'vicuna'

        self.processor = LlavaProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(**self.params)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")

    async def chat_with_images(self, messages: list[Message], max_tokens: int) -> str:
                               
        images, prompt = await prompt_from_messages(messages, self.format)
        inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        
        if self.format in ['llama2', 'mistral']:
            idx = answer.rfind('[/INST]') + len('[/INST]') + 1 #+ len(images)
            return answer[idx:]
        elif self.format == 'vicuna':
            idx = answer.rfind('ASSISTANT:') + len('ASSISTANT:') + 1 #+ len(images)
            return answer[idx:]
        elif self.format == 'chatml':
            idx = answer.rfind('<|im_user|>assistant\n') + len('<|im_user|>assistant\n') + 1 #+ len(images)
            end_idx = answer.rfind('<|im_end|>')
            return answer[idx:end_idx]

        return answer