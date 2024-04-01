
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import VisionQnABase

class VisionQnA(VisionQnABase):
    model_name: str = "moondream2"
    revision: str = '2024-03-13'
    
    def __init__(self, model_id: str, device: str, extra_params = {}):
        if device == 'auto':
            device = self.select_device()

        params = {
            'pretrained_model_name_or_path': model_id,
            'trust_remote_code': True,
            'revision': self.revision,
            'torch_dtype': torch.float32 if device == 'cpu' else torch.float16,
        }
        params.update(extra_params)

        self.model = AutoModelForCausalLM.from_pretrained(**params).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    async def single_question(self, image_url: str, prompt: str) -> str:
        image = await self.url_to_image(image_url)
        encoded_image = self.model.encode_image(image)
        return self.model.answer_question(encoded_image, prompt, self.tokenizer)
