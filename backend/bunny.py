from transformers import AutoTokenizer, AutoModelForCausalLM, logging

from vision_qna import *

import warnings
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# "BAAI/Bunny-Llama-3-8B-V"

class VisionQnA(VisionQnABase):
    model_name: str = "bunny"
    format: str = "vicuna"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        torch.set_default_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).to(self.device).eval()
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        text_chunks = [self.tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(self.model.device)

        image_tensor = self.model.process_images(images, self.model.config).to(dtype=self.model.dtype, device=self.model.device)

        params = self.get_generation_params(request)
        output_ids = self.model.generate(input_ids, images=image_tensor, **params)

        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return response
