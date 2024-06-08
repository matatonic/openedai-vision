from threading import Thread
from transformers import AutoTokenizer, AutoProcessor, logging
from dragonfly.models.modeling_dragonfly import DragonflyForCausalLM
from dragonfly.models.processing_dragonfly import DragonflyProcessor

import warnings
# disable some warnings
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

from vision_qna import *

# togethercomputer/Llama-3-8B-Dragonfly-v1
# togethercomputer/Llama-3-8B-Dragonfly-Med-v1

class VisionQnA(VisionQnABase):
    model_name: str = "dragonfly"
    format: str = 'llama3'
    vision_layers: List[str] = ['image_encoder', 'vision_model', 'encoder', 'mpl', 'vision_embed_tokens']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        del self.params['trust_remote_code']

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = DragonflyProcessor(image_processor=clip_processor.image_processor, tokenizer=self.tokenizer, image_encoding_style="llava-hd")

        self.model = DragonflyForCausalLM.from_pretrained(**self.params)

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(dtype=self.dtype, device=self.device)

        self.eos_id = "<|eot_id|>"
        self.eos_token_id = self.tokenizer.encode(self.eos_id, add_special_tokens=False)
    
        print(f"Loaded {model_id} on device: {self.model.device} with dtype: {self.model.dtype}")

    async def stream_chat_with_images(self, request: ImageChatRequest):
        images, prompt = await llama3_prompt_from_messages(request.messages, img_tok='')

        inputs = self.processor(text=[prompt], images=images, max_length=2048, return_tensors="pt", is_generate=True).to(device=self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=False, skip_prompt=True)

        default_params = {
            'max_new_tokens': 1024,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.eos_token_id[0],
        }

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            **inputs,
            **params,
            streamer=streamer,
        )

        t = Thread(target=self.model.generate, kwargs=generation_kwargs)
        t.start()

        for new_text in streamer:
            end = new_text.find(self.eos_id)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
