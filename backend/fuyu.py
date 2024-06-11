from transformers import FuyuProcessor, FuyuForCausalLM

from vision_qna import *

# "adept/fuyu-8b"

class VisionQnA(VisionQnABase):
    model_name: str = "fuyu"
    format: str = "fuyu"
    vision_layers: List[str] = ["vision_embed_tokens"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)
        
        del self.params['trust_remote_code'] # not needed.

        self.processor = FuyuProcessor.from_pretrained(model_id)
        self.model = FuyuForCausalLM.from_pretrained(**self.params)

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        inputs = self.processor(text=prompt, images=images[0] if images else None, return_tensors="pt").to(self.model.device)

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
