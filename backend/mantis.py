from mantis.models.mllava import chat_mllava, MLlavaProcessor, LlavaForConditionalGeneration
from mantis.models.mfuyu import MFuyuForCausalLM, MFuyuProcessor

from vision_qna import *

class VisionQnA(VisionQnABase):
    format: str = 'internal'
    model_name: str = "mantis"
    vision_layers: List[str] = ["vision_tower", "multi_modal_projector"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)
        
        del self.params['trust_remote_code']

        if '-fuyu' in model_id.lower():
            self.processor = MFuyuProcessor.from_pretrained(model_id)
            self.model = MFuyuForCausalLM.from_pretrained(**self.params)
        else:
            self.processor = MLlavaProcessor.from_pretrained(model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(**self.params)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        prompt, history, images, system = await prompt_history_images_system_from_messages(request.messages, img_tok = "<image>", url_handler = url_to_image)

        default_params = {
            'num_beams': 1,
            'do_sample': False,
        }

        params = self.get_generation_params(request, default_params)

        response, history = chat_mllava(prompt, images if images else None, self.model, self.processor, history=history if history else None, **params)

        return response

