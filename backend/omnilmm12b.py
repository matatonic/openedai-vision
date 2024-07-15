from transformers import AutoTokenizer, AutoModel

from vision_qna import *

# openbmb/OmniLMM-12B

class VisionQnA(VisionQnABase):
    format: str = 'internal'
    model_name: str = "omnilmm12b"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=2048, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).to(dtype=self.params['torch_dtype']).eval()
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        # 3B
        image = None
        msgs = []

        for m in request.messages:
            if m.role == 'user':
                for c in m.content:
                    if c.type == 'image_url':
                        image = await url_to_image(c.image_url.url)
                    if c.type == 'text':
                        msgs.extend([{ 'role': 'user', 'content': c.text }])
            elif m.role == 'assistant':
                for c in m.content:
                    if c.type == 'text':
                        msgs.extend([{ 'role': 'assistant', 'content': c.text }])

        params = self.get_generation_params(request)

        answer, context, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **params,
        )

        return answer
    