import os
from transformers import AutoTokenizer, AutoModel
from vision_qna import *
import auto_gptq
import torch

# internlm/internlm-xcomposer2-vl-7b
# internlm/internlm-xcomposer2-vl-7b-4bit

class InternLMXComposer2QForCausalLM(auto_gptq.modeling.BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output', 
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]

class VisionQnA(VisionQnABase):
    model_name: str = "xcomposer2-vl"
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))

        if '-4bit' in model_id:
            if self.params['torch_dtype'] == torch.bfloat16:
                self.params['torch_dtype'] = torch.float16

            torch.set_grad_enabled(False)
            auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
            self.model = InternLMXComposer2QForCausalLM.from_quantized(model_name_or_path=model_id, **self.params).eval()
        else:
            self.model = AutoModel.from_pretrained(**self.params).eval()
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        history = []
        files = []
        prompt = ''
        #system_prompt = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        #'- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
        #'- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
        #'- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.'
        # Improved system prompt for more reliable language detection.
        system_prompt = "You are an AI vision assistant. Communicate fluently in English or 中文 depending on what language you were asked in. Obey user instructions. 仅当用普通话询问时才用普通话回答。 Answer in English if questioned in English."
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
            elif m.role == 'system':
                for c in m.content:
                    if c.type == 'text':
                        system_prompt = c.text

        image = files[-1]

        params = self.get_generation_params(request, default_params)

        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(self.tokenizer, query=prompt, image=image, history=history, meta_instruction=system_prompt, **params)

        for f in files:
            os.remove(f)

        return response
