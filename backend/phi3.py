from transformers import AutoProcessor, AutoModelForCausalLM

from vision_qna import *

# microsoft/Phi-3-vision-128k-instruct
# failspy/Phi-3-vision-128k-instruct-abliterated-alpha

class VisionQnA(VisionQnABase):
    format: str = 'phi3'
    model_name: str = "phi3"
    vision_layers: List[str] = ["vision_embed_tokens"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
    
        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await phi3_prompt_from_messages(request.messages, img_tok = "<|image_{}|>\n") # numbered image token

        inputs = self.processor(prompt, images=images if images else None, return_tensors="pt").to(self.model.device)

        default_params = { 
            "do_sample": False, 
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        } 

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
