from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import os
from vision_qna import *

# Qwen/Qwen2-VL-2B-Instruct-AWQ
# Qwen/Qwen2-VL-2B-Instruct
# Qwen/Qwen2-VL-7B-Instruct-AWQ
# Qwen/Qwen2-VL-7B-Instruct
# Qwen/Qwen2-VL-72B-Instruct-AWQ
# X Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4
# X Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8
# X Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4
# X Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8

class VisionQnA(VisionQnABase):
    model_name: str = "qwen2-vl"
    format: 'chatml'
    vision_layers: List[str] = ['visual']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if 'awq' in model_id.lower() and self.dtype == torch.bfloat16:
            self.dtype = self.params['torch_dtype'] = torch.float16  # recommended

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        del self.params['trust_remote_code']

        if model_id == 'Qwen/Qwen2-VL-7B-Instruct-AWQ':
            self.params['revision'] = '9d72ae62396aaa1817b006e07ddbbd121024f50d' # re: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-AWQ/discussions/4

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(**self.params).eval()

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        # image_tag = '<|vision_start|><|image_pad|><|vision_end|>'

        messages = []

        for m in request.messages:
            if m.role == 'user':
                msg = { 'role': m.role, 'content': [] }
                for c in m.content:
                    if c.type == 'image_url':
                        # hack around https://github.com/QwenLM/Qwen2-VL/issues/202'
                        if c.image_url.url.startswith('data:image'):
                            parts = c.image_url.url.split(';')
                            if parts[1].startswith('charset='):
                                c.image_url.url = parts[0] + ';' + parts[2]

                        msg['content'].extend([{'type': c.type, 'image': c.image_url.url}])
                    elif c.type == 'text':
                        msg['content'].extend([{'type': c.type, 'text': c.text}])
                    elif c.type == 'video': # not likely to work.
                        msg['content'].extend([{'type': c.type, 'video': c.image_url.url}])
            else:
                ctext = "".join([c.text for c in m.content]) # fix for multiple system prompt contents #19
                msg = { 'role': m.role, 'content': [{ 'type': 'text', 'text': ctext }] }

            messages.extend([msg])

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        params = self.get_generation_params(request, default_params={})

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.processor.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.processor.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
