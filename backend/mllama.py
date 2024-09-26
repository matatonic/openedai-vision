from transformers import MllamaForConditionalGeneration, AutoProcessor

from vision_qna import *

# meta-llama/Llama-3.2-11B-Vision-Instruct
# meta-llama/Llama-3.2-90B-Vision-Instruct

class VisionQnA(VisionQnABase):
    model_name: str = "mllama"
    format: str = "llama3"
    visual_layers: List[str] = ['vision_model', 'multi_modal_projector']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        del self.params['trust_remote_code']

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MllamaForConditionalGeneration.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await llama3_prompt_from_messages(request.messages, img_tok = "<|image|>")

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = "<|image|>" + prompt

        inputs = self.processor(images, prompt, return_tensors="pt").to(self.model.device)

        default_params = dict(do_sample=True)

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.processor, generation_kwargs=generation_kwargs):
            yield new_text
