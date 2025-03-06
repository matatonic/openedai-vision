from transformers import AriaProcessor, AriaForConditionalGeneration

from vision_qna import *

# rhymes-ai/Aria

class VisionQnA(VisionQnABase):
    model_name: str = "aria" # idefics3_vision
    format: str = "chatml"
    visual_layers: List[str] = ["vision_tower", "multi_modal_projector"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.processor = AriaProcessor.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AriaForConditionalGeneration.from_pretrained(**self.params).eval()

        self.eos_token = '<|im_end|>'

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await chatml_prompt_from_messages(request.messages, img_tok = "<fim_prefix><|img|><fim_suffix>")

        prompt = prompt.replace("<fim_suffix><fim_prefix>", "<fim_suffix>\n<fim_prefix>")#.replace('<|im_end|>', '<|im_end|>\n')

        if len(images) < 1:
            prompt = "<fim_prefix><|img|><fim_suffix>" + prompt
            images = [await url_to_image(transparent_pixel_url)]

        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        inputs = inputs.to(self.model.device)

        default_params = {
            'max_new_tokens': 500,
            'do_sample': False,
#            'temperature': 0.9, # random test failures, ex. OCR
            'stop_strings': [self.eos_token],
        }

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            tokenizer=self.processor.tokenizer,
            **inputs,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.processor.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
