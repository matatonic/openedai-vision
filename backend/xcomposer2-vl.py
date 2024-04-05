import os
from transformers import AutoTokenizer, AutoModel

from vision_qna import *

# internlm/internlm-xcomposer2-vl-7b

class VisionQnA(VisionQnABase):
    model_name: str = "xcomposer2-vl"
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, messages: list[Message], max_tokens: int) -> str:
        history = []
        files = []
        prompt = ''

        for m in messages:
            if m.role == 'user':
                p = ''
                for c in m.content:
                    if c.type == 'image_url':
                        filename = await url_to_file(c.image_url.url)
                        p = '<ImageHere>' + p
                        files.extend([filename])
                    if c.type == 'text':
                        p += c.text

                prompt += p
            elif m.role == 'assistant':
                for c in m.content:
                    if c.type == 'text':
                        history.extend([(prompt, c.text)])
                        prompt = ''


        image = files[-1]
        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(self.tokenizer, query=prompt, image=image, history=history, do_sample=False, max_new_tokens=max_tokens)

        for f in files:
            os.remove(f)

        return response
