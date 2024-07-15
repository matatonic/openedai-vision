import os
from transformers import AutoTokenizer, AutoModel, logging
from vision_qna import *
import auto_gptq
import torch

import warnings
# disable some warnings
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# internlm/internlm-xcomposer2-7b
# --4bit | TypeError: Linear4bit.forward() takes 2 positional arguments but 3 were given
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

class VisionQnA(VisionQnABase):
    format: str = 'internal'
    model_name: str = "xcomposer2"
    vision_layers: List[str] = ['vit', 'vision_proj']

    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        if '-4bit' in model_id:
            if self.params['torch_dtype'] == torch.bfloat16:
                self.dtype = self.params['torch_dtype'] = torch.float16 # fix bugs.
                torch.set_default_dtype(self.dtype)
            
            # XXX TODO: use_marlin=True
            auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
            self.model = InternLMXComposer2QForCausalLM.from_quantized(model_name_or_path=model_id, **self.params)
        else:
            torch.set_default_dtype(self.dtype)
            self.model = AutoModel.from_pretrained(**self.params).eval()

            # bitsandbytes already moves the model to the device, so we don't need to do it again.
            if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
                self.model = self.model.to(self.device)
    
        self.eos_token = '[UNUSED_TOKEN_145]'
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        query, history, images, meta_instruction = await prompt_history_images_system_from_messages(
            request.messages, img_tok='<ImageHere>', url_handler=url_to_image)
        
        if meta_instruction is None:
            #meta_instruction = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
            #'- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
            #'- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.'
            # This only works if the input is in English, Chinese input still receives Chinese output.
            meta_instruction = "You are an AI visual assistant. Communicate in English. Do what the user instructs."

        # this may be the only difference with, -vl, images in PIL instead of files.
        image = torch.stack([self.model.vis_processor(image) for image in images]) if images else None

        if image is None:
            inputs = self.model.build_inputs(self.tokenizer, query, history, meta_instruction)
            im_mask = torch.zeros(inputs['input_ids'].shape[:2]).cuda().bool()
        else:
            image = self.model.encode_img(image)
            inputs, im_mask = self.model.interleav_wrap_chat(self.tokenizer, query, image, history, meta_instruction)
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items() if torch.is_tensor(v)
        }
        inputs['im_mask'] = im_mask

        default_params = {
            "temperature": 1.0,
            "top_p": 0.8,
            'do_sample': True,
        }

        params = self.get_generation_params(request, default_params)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        try:
            def wrapper(**kwargs):
                with torch.cuda.amp.autocast():
                    _ = self.model.generate(**kwargs)

            for new_text in threaded_streaming_generator(generate=wrapper, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
                end = new_text.find(self.eos_token)
                if end == -1:
                    yield new_text
                else:
                    yield new_text[:end]
                    break

        except Exception as e:
            logger.error(e)
            # raise
