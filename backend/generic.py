from transformers import AutoTokenizer, AutoModel

from vision_qna import *

class VisionQnA(VisionQnABase):
    model_name: str = "generic"
    format: str = "generic"
    visual_layers: List[str] = []
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        inputs = self.tokenizer(prompt, images=images, return_tensors="pt").to(self.model.device)

        default_params = {
            'do_sample': False,
        }

        params = self.get_generation_params(request, default_params=default_params)

        output = self.model.generate(**inputs, **params)
        response = self.tokenizer.decode(output[0][inputs.input_ids.size(1):].cpu(), skip_special_tokens=True)

        return response

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        inputs = self.tokenizer(prompt, images=images, return_tensors="pt").to(self.model.device)

        default_params = {
            'do_sample': False,
        }

        params = self.get_generation_params(request, default_params=default_params)

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
