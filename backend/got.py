from transformers import AutoTokenizer, AutoModel

from vision_qna import *

# ucaslcl/GOT-OCR2_0 XXX
# stepfun-ai/GOT-OCR2_0

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

class VisionQnA(VisionQnABase):
    model_name: str = "got"
    format: str = "custom"
    visual_layers: List[str] = ['vision_tower_high', 'mm_projector_vary']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        try:
            image = None
            for m in reversed(request.messages):
                for c in m.content:
                    if c.type == 'image_url':
                        image = await url_to_file(c.image_url.url)
                        break

            response = self.model.chat(self.tokenizer, image, ocr_type='ocr') # TODO: support format and maybe convert to markdown?

            return response
        finally:
            if image:
                os.remove(image)
