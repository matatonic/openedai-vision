from transformers import AutoTokenizer, AutoModel

from vision_qna import *

# openbmb/MiniCPM-V 
# aka OmniLMM-3B

class VisionQnA(VisionQnABase):
    model_name: str = "omnilmm3b"
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)

        # bugs with 4bit, RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).to(dtype=self.params['torch_dtype']).eval()
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, messages: list[Message], max_tokens: int) -> str:
        # 3B
        image = None
        msgs = []

        for m in messages:
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

        answer, context, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens
        )

        return answer