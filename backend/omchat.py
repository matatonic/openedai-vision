from transformers import AutoModel, AutoProcessor, AutoTokenizer

from vision_qna import *

# omlab/omchat-v2.0-13B-single-beta_hf

class VisionQnA(VisionQnABase):
    model_name: str = "omchat"
    format: str = "chatml"
    visual_layers: List[str] = ['vision_tower', 'multi_modal_projector']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if self.params['torch_dtype'] == torch.bfloat16:
            self.dtype = self.params['torch_dtype'] = torch.float16

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        # XXX bug fix, model seems to alter the config after first generation
        self.eos_token_id = self.model.generation_config.eos_token_id

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await chatml_prompt_from_messages(request.messages, img_tok='<image>')

        if len(images) < 1:
            images = None

        inputs = self.processor(prompt, images=images, return_tensors="pt").to(self.model.device)

        default_params = dict(
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

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
