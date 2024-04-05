from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import os
import uuid
from datauri import DataURI
from vision_qna import *

# "Qwen/Qwen-VL-Chat" # 13GB
# "Qwen/Qwen-VL-Chat-4bit" # TODO: auto-gptq

class VisionQnA(VisionQnABase):
    model_name: str = "qwen-vl"
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")

    async def chat_with_images(self, messages: list[Message], max_tokens: int) -> str:

        history = []
        files = []
        prompt = ''
        image_url = None

        for m in messages:
            if m.role == 'user':
                for c in m.content:
                    if c.type == 'image_url':
                        # XXX if data: url, save to local file.
                        if c.image_url.url.startswith('data:'):
                            # secure temp filename
                            filename = f"/tmp/{uuid.uuid4()}"
                            with open(filename, 'wb') as f:
                                f.write(DataURI(c.image_url.url).data)
                                files.extend([filename])
                            image_url = filename
                        else:
                            image_url = c.image_url.url
                    if c.type == 'text':
                        prompt = c.text
                
            elif m.role == 'assistant':
                for c in m.content:
                    if c.type == 'text':
                        history.extend([(prompt, c.text)])
                        prompt = ''

        # 1st dialogue turn
        query = self.tokenizer.from_list_format([
            {'image': image_url},
            {'text': prompt},
        ])

        answer, history = self.model.chat(self.tokenizer, query=query, history=history)

        for f in files:
            os.remove(f)

        return answer
