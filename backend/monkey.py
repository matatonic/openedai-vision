import os
import uuid
from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

# echo840/Monkey

class VisionQnA(VisionQnABase):
    model_name: str = "monkey"
    format: str = 'qwen' # phi15-ish
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)

         # XXX currently bugged https://huggingface.co/echo840/Monkey/discussions/4
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))

        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        files = []
        prompt = ''
    
        for m in request.messages:
            if m.role == 'user':
                p = ''
                for c in m.content:
                    if c.type == 'image_url':
                        filename = await url_to_file(c.image_url.url)
                        p = '<img>' + filename + '</img> ' + p
                    if c.type == 'text':
                        p += f"{c.text}\n\n" # Question:
                prompt += p
            elif m.role == 'assistant':
                for c in m.content:
                    if c.type == 'text':
                        prompt += f"Answer: {c.text}\n\n"

        prompt += "Answer:"

        input_ids = self.tokenizer(prompt, return_tensors='pt', padding='longest')

        attention_mask = input_ids.attention_mask.to(self.model.device)
        input_ids = input_ids.input_ids.to(self.model.device)

        params = self.get_generation_params(request)

        pred = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=self.tokenizer.eod_id,
            eos_token_id=self.tokenizer.eod_id,
            **params,
        )
        response = self.tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()

        for f in files:
            os.remove(f)

        return response
