from transformers import LlavaProcessor, LlavaForConditionalGeneration
from vision_qna import *

#
# llava-hf/bakLlava-v1-hf # llama2
# llava-hf/llava-1.5-7b-hf # vicuna
# llava-hf/llava-1.5-13b-hf # vicuna
# Doesn't support execution without images

class VisionQnA(VisionQnABase):
    model_name: str = "llava"
    format: str = 'vicuna'
    vision_layers: List[str] = ["vision_model", "vision_tower", "multi_modal_projector"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        del self.params['trust_remote_code']
        
        self.processor = LlavaProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(**self.params)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
                               
        images, prompt = await prompt_from_messages(request.messages, self.format)

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = "<image>\n" + prompt

        inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)

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
