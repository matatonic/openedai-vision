from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from vision_qna import *

# model_id = "llava-hf/llava-v1.6-mistral-7b-hf" # llama2
# model_id = "llava-hf/llava-v1.6-34b-hf" # chatml
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf" # vicuna
# model_id = "llava-hf/llava-v1.6-vicuna-7b-hf" #  vicuna

class VisionQnA(VisionQnABase):
    model_name: str = "llavanext"
    format: str = 'llama2'
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)

        if not format:
            if 'mistral' in model_id:
                self.format = 'llama2'
            elif 'vicuna' in model_id:
                self.format = 'vicuna'
            elif 'v1.6-34b' in model_id:
                self.format = 'chatml'

        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(**self.params)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")

    async def chat_with_images(self, messages: list[Message], max_tokens: int) -> str:
                               
        images, prompt = await prompt_from_messages(messages, self.format)
        inputs = self.processor(prompt, images, return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        
        if self.format in ['llama2', 'mistral']:
            idx = answer.rfind('[/INST]') + len('[/INST]') + 1 #+ len(images)
            return answer[idx:]
        elif self.format == 'vicuna':
            idx = answer.rfind('ASSISTANT:') + len('ASSISTANT:') + 1 #+ len(images)
            return answer[idx:]
        elif self.format == 'chatml':
            # XXX This is broken with the 34b, extra spaces in the tokenizer
            # XXX You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
            idx = answer.rfind('<|im_start|>assistant\n') + len('<|im_start|>assistant\n') + 1 #+ len(images)
            end_idx = answer.rfind('<|im_end|>')
            return answer[idx:end_idx]

        return answer
