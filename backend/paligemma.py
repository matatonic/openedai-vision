# "google/paligemma2-3b-ft-docci-448"
# "google/paligemma2-10b-ft-docci-448"
# "google/paligemma2-28b-pt-896" - pretrain

from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from vision_qna import *

class VisionQnA(VisionQnABase):
    model_name: str = "paligemma2"
    format: str = "gemma" # doesn't seem to actually be instruction trained
    visual_layers: List[str] = ["vision_tower", "multi_modal_projector"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        for i in ['trust_remote_code']:
            del self.params[i]

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(**self.params).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = "<image>\n" + prompt

        # Instruct the model to create a caption in English
        #prompt = "caption en"
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(dtype=self.dtype, device=self.device)

        default_params = {
            'do_sample': False,
#            'eos_token_id': self.processor.tokenizer.eos_token_id,
#            'pad_token_id': self.processor.tokenizer.eos_token_id,
        }

        params = self.get_generation_params(request, default_params=default_params)

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
