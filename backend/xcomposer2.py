import os
from transformers import AutoTokenizer, AutoModel
from vision_qna import *
import auto_gptq
import torch

import transformers
import warnings
# disable some warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# internlm/internlm-xcomposer2-7b
# internlm/internlm-xcomposer2-7b-4bit # pretty bad.

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

# internlm/internlm-xcomposer2-7b

class VisionQnA(VisionQnABase):
    model_name: str = "xcomposer2"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        if '-4bit' in model_id:
            if self.params['torch_dtype'] == torch.bfloat16:
                self.params['torch_dtype'] = torch.float16 # fix bugs.
            
            # XXX TODO: use_marlin=True
            torch.set_grad_enabled(False)
            auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
            self.model = InternLMXComposer2QForCausalLM.from_quantized(model_name_or_path=model_id, **self.params)
        else:
            self.model = AutoModel.from_pretrained(**self.params).eval()

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        prompt, history, images, system_prompt = await prompt_history_images_system_from_messages(
            request.messages, img_tok='<ImageHere>', url_handler=url_to_image)
        
        if system_prompt is None:
            #system_prompt = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
            #'- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
            #'- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.'
            # This only works if the input is in English, Chinese input still receives Chinese output.
            system_prompt = "You are an AI visual assistant. Communicate in English. Do what the user instructs."

        images = [self.model.vis_processor(image) for image in images]

        default_params = {
            "temperature": 1.0,
            "top_p": 0.8,
            'do_sample': True,
        }

        params = self.get_generation_params(request, default_params)

        image = torch.stack(images)
        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(self.tokenizer, query=prompt, image=image, history=history, meta_instruction=system_prompt, **params)


        return response
