from transformers import AutoTokenizer, AutoModel

from vision_qna import *

# openbmb/MiniCPM-Llama3-V-2_5
# openbmb/MiniCPM-V-2
# openbmb/MiniCPM-V aka OmniLMM-3B

class VisionQnA(VisionQnABase):
    model_name: str = "minicpm"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        # bugs with 4bit, RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval().to(dtype=torch.float16)
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        # 3B
        image = None
        msgs = []
        #system_prompt = ''
        default_sampling_params = {
            'do_sample': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.6,
        }

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
            elif m.role == 'system':
                for c in m.content:
                    if c.type == 'text':
                        msgs.extend([{ 'role': 'user', 'content': c.text }, { 'role': 'assistant', 'content': "OK" }])  # fake system prompt

        # default uses num_beams: 3, but if sampling is requested, switch the defaults.
        params = self.get_generation_params(request)
        if params.get('do_sample', False):
            params = self.get_generation_params(request, default_sampling_params)

        answer = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=params.get('do_sample', False),
            **params,
        )

        if not isinstance(answer, str):
            answer = answer[0]

        return answer
