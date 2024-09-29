from transformers import AutoProcessor, AutoModel

from vision_qna import *

class VisionQnA(VisionQnABase):
    model_name: str = "generic"
    format: str = "generic"
    visual_layers: List[str] = []
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.model.device)

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

    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.model.device)

        default_params = {
            'do_sample': False,
#            'eos_token_id': self.processor.tokenizer.eos_token_id,
#            'pad_token_id': self.processor.tokenizer.eos_token_id,
        }

        params = self.get_generation_params(request, default_params=default_params)


        tps_start = time.time()
        output = self.model.generate(**inputs, **params)
        out_tokens = output[0][inputs.input_ids.size(1):].cpu()
        logger.info(f"Generated {len(out_tokens)} tokens at {len(out_tokens) / (time.time() - tps_start):0.2f} T/s")
        response = self.processor.tokenizer.decode(out_tokens, skip_special_tokens=True)

        return response

