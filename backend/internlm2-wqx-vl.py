import os
from math import ceil
import warnings
import torch
from transformers import AutoTokenizer, AutoModel, logging
from torchvision import transforms
from huggingface_hub import snapshot_download


from vision_qna import *

#logging.set_verbosity_error()
#warnings.filterwarnings('ignore')

# internlm/internlm2-wqx-vl-20b

# --4bit:
# Linear4bit.forward() takes 2 positional arguments but 3 were given

class VisionQnA(VisionQnABase):
    model_name: str = "internlm2-wqx-vl"
    format: str = "chatml"
    vision_layers: List[str] = ['vit', 'vision_proj', 'vision_tower']

    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)
        
        #torch.set_default_dtype(self.dtype)

        self.params['pretrained_model_name_or_path'] = model_id = snapshot_download(model_id)

        #self.max_tiles = extra_params.get('max_tiles', MAX_TILES)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
            self.model = self.model.to(self.device)

        self.eos_token = '<|im_end|>' # [UNUSED_TOKEN_145]
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

        self.loaded_banner()
    

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await chatml_prompt_from_messages(request.messages, img_tok = "")

        system_default = ("You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."),

        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items() if torch.is_tensor(v)
        }

        if images:
            images = self.model.vis_processor(images[-1]).unsqueeze(0).to(self.model.device)

            # XXX server-1  |     sub_img = img.reshape(1,3,H//560,560,W//560,560).permute(0,2,4,1,3,5).reshape(-1,3,560,560).contiguous()
            # Ex. RuntimeError: shape '[1, 3, 1, 560, 1, 560]' is invalid for input of size 1224216
            img_embeds, img_split = self.model.vit([images], self.model.plora_glb_GN, self.model.plora_sub_GN)

            img_embeds = self.model.vision_proj(img_embeds)
            inputs['img_embeds'] = img_embeds

        default_params = {
            #'num_beams': 3,
            #'do_sample': False,
            "temperature": 0.8,
            "top_p": 0.8,
            'do_sample': True,
            'repetition_penalty': 1.005,
            'eos_token_id': [ self.tokenizer.eos_token_id, self.eos_token_id ], # also add end-of-assistant token in eos token id to avoid unnecessary generation
        }
        params = self.get_generation_params(request, default_params)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

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

