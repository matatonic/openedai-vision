import os
from transformers import AutoTokenizer, AutoModel

from vision_qna import *

# internlm/internlm-xcomposer2-7b

class VisionQnA(VisionQnABase):
    model_name: str = "xcomposer2"
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        history = []
        images = []
        prompt = ''
        #system_prompt = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        #'- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
        #'- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.'
        # This only works if the input is in English, Chinese input still receives Chinese output.
        system_prompt = "You are an AI visual assistant. Communicate in English. Do what the user instructs."
        default_params = {
            "temperature": 1.0,
            "top_p": 0.8,
            'do_sample': True,
        }

        for m in request.messages:
            if m.role == 'user':
                p = ''
                for c in m.content:
                    if c.type == 'image_url':
                        image = await url_to_image(c.image_url.url)
                        image = self.model.vis_processor(image)
                        images.extend([image])
                        p = '<ImageHere>' + p
                    if c.type == 'text':
                        p += c.text

                prompt += p
            elif m.role == 'assistant':
                for c in m.content:
                    if c.type == 'text':
                        history.extend([(prompt, c.text)])
                        prompt = ''
            elif m.role == 'system':
                for c in m.content:
                    if c.type == 'text':
                        system_prompt = c.text

        params = self.get_generation_params(request, default_params)

        image = torch.stack(images)
        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(self.tokenizer, query=prompt, image=image, history=history, meta_instruction=system_prompt, **params)


        return response
