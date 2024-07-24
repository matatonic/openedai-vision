from transformers import AutoTokenizer, AutoModel

from vision_qna import *

# openbmb/MiniCPM-Llama3-V-2_5 # broken after 45387f99a455e11801b78a0b24811856688e0c8b
# openbmb/MiniCPM-V-2  - 4bit broken
# openbmb/MiniCPM-V aka OmniLMM-3B

class VisionQnA(VisionQnABase):
    format: str = 'internal'
    model_name: str = "minicpm"
    vision_layers: List[str] = ["resampler", "vpm"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        # I wish there was a better way to do this... 
        if model_id == 'openbmb/MiniCPM-Llama3-V-2_5':
            self.revision = '45387f99a455e11801b78a0b24811856688e0c8b'

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
            self.model = self.model.to(dtype=self.params['torch_dtype'], device=self.device)
    
        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        image = None
        msgs = []
        system_prompt = None

        for m in request.messages:
            if m.role == 'user':
                for c in m.content:
                    if c.type == 'image_url':
                        image = await url_to_image(c.image_url.url)
            for c in m.content:
                if c.type == 'text':
                    if m.role == 'system':
                        system_prompt = c.text
                    else:
                        msgs.extend([{ 'role': m.role, 'content': c.text }])

        if image is None:
            image = await url_to_image(black_pixel_url)

        # default uses num_beams: 3, but if streaming/sampling is requested, switch the defaults.
        default_sampling_params = {
            'do_sample': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.6,
        }
        params = self.get_generation_params(request, default_sampling_params)

        with torch.cuda.amp.autocast():
            answer = self.model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=self.tokenizer,
                sampling=True,
                system_prompt=system_prompt,
                stream=True,
                **params,
            )

        if isinstance(answer, str):
            answer = [answer]

        for new_text in answer:
            if isinstance(new_text, str):
                yield new_text

