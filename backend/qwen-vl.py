from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import os
import uuid
from datauri import DataURI
from vision_qna import *

# "Qwen/Qwen-VL-Chat" # 13GB
# "Qwen/Qwen-VL-Chat-int4" # 11GB (bad, bugs)

class VisionQnA(VisionQnABase):
    model_name: str = "qwen-vl"
    format: 'chatml'
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")

    async def chat_with_images(self, request: ImageChatRequest) -> str:
        prompt, history, files, system_prompt = await prompt_history_images_system_from_messages(
            request.messages, img_tok='', url_handler=url_to_file)

        if system_prompt is None:
            system_prompt =  "You are an helpful assistant."
            
        # 1st dialogue turn
        query = self.tokenizer.from_list_format([
            {'image': files[-1] if files else []},
            {'text': prompt},
        ])

        default_params = {
            'top_p': 0.3,
        }

        params = self.get_generation_params(request)

        answer, history = self.model.chat(self.tokenizer, query=query, history=history, system=system_prompt, **params)

        for f in files:
            os.remove(f)

        return answer
