from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from vision_qna import *

# llava-hf/llava-v1.6-34b-hf # chatml
# llava-hf/llava-v1.6-vicuna-13b-hf # vicuna
# llava-hf/llava-v1.6-vicuna-7b-hf #  vicuna
# llava-hf/llava-v1.6-mistral-7b-hf # llama2
# tiiuae/falcon-11B-vlm # falcon

# llavanext doesn't support generation without images

class VisionQnA(VisionQnABase):
    model_name: str = "llavanext"
    format: str = 'llama2'
    vision_layers: List[str] = ["vision_model", "vision_tower", "multi_modal_projector"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        del self.params['trust_remote_code']

        use_fast = 'mistral' in model_id or 'falcon' in model_id
        self.processor = LlavaNextProcessor.from_pretrained(model_id, use_fast=use_fast)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(**self.params)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = "<image>\n" + prompt

        inputs = self.processor(prompt, images, return_tensors="pt").to(self.model.device)

        default_params = dict(
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        params = self.get_generation_params(request, default_params)

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
