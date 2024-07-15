from llava.conversation import conv_templates
from llava.mm_utils import (
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model import LlavaLlamaForCausalLM
from llava.model.constants import IMAGE_TOKEN_INDEX

from transformers import AutoTokenizer


from vision_qna import *

# 01-ai/Yi-VL-34B
# 01-ai/Yi-VL-6B

class VisionQnA(VisionQnABase):
    model_name: str = "qwen-vl"
    format: str = 'chatml'
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = LlavaLlamaForCausalLM.from_pretrained(**self.params)

        image_processor = None
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.vision_tower = self.model.get_vision_tower()

        if not self.vision_tower.is_loaded:
            self.vision_tower.load_model()
        self.vision_tower.to(device=self.model.device, dtype=self.model.dtype)
        self.image_processor = self.vision_tower.image_processor

        if hasattr(self.model.config, "max_sequence_length"):
            context_len = self.model.config.max_sequence_length
        else:
            context_len = 2048

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        # XXX
        images, prompt = await prompt_from_messages(request.messages, self.format)

        encoded_images = self.model.encode_image(images)
        inputs = self.tokenizer(prompt, encoded_images, return_tensors="pt")

        params = self.get_generation_params(request)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
