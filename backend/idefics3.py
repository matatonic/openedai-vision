from transformers import AutoProcessor, AutoModelForVision2Seq

from vision_qna import *

# HuggingFaceTB/SmolVLM-Instruct
# HuggingFaceM4/Idefics3-8B-Llama3

class VisionQnA(VisionQnABase):
    model_name: str = "idefics3"
    format: str = "internal"
    visual_layers: List[str] = ["vision_model"]

    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, messages = await images_hfmessages_from_messages(request.messages)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = "<image>\n" + prompt

        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)

        params = self.get_generation_params(request)

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
