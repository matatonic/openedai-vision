from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

from vision_qna import *

# allenai/MolmoE-1B-0924 XXX problems with performance and RAM usage
# allenai/Molmo-7B-D-0924 # faster
# allenai/Molmo-7B-O-0924
# allenai/Molmo-72B-0924
# SeanScripts/Molmo-72B-0924-nf4
# cyan2k/molmo-7B-D-bnb-4bit XXX needs tensorflow-cpu
# cyan2k/molmo-7B-O-bnb-4bit XXX needs tensorflow-cpu

class VisionQnA(VisionQnABase):
    model_name: str = "molmo"
    format: str = "chatml"
    visual_layers: List[str] = ['vision_backbone']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        #self.dtype = self.params['torch_dtype'] = 'auto' # torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False), torch_dtype=self.params['torch_dtype'], device_map=self.params['device_map'])
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.eos_token_id = self.processor.tokenizer.encode(self.processor.tokenizer.eos_token)[0]

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await chatml_prompt_from_messages(request.messages, img_tok = "<|image|>")

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = "<|image|>" + prompt

        # process the image and text
        inputs = self.processor.process(
            images=images,
            text=prompt,
        )
        
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        default_params = dict(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id
        )

        params = self.get_generation_params(request, default_params)

        generation_kwargs = dict(
            batch=inputs,
            generation_config=GenerationConfig(**params)
        )

        def wrapper(**kwargs):
            with torch.amp.autocast('cuda', dtype=self.dtype):
                _ = self.model.generate_from_batch(**kwargs)

        for new_text in threaded_streaming_generator(generate=wrapper, tokenizer=self.processor.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.processor.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
