from math import ceil
import warnings
import torch
from transformers import AutoTokenizer, AutoModel, logging

from vision_qna import *

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# "internlm/internlm-xcomposer2-4khd-7b"
MAX_TILES = 40

def calc_hd(image, max_num=MAX_TILES):
    # Not sure if this is correct, but there are no instructions for how to set it
    img = Image.open(image)
    width, height = img.size
    del img

    return min(ceil(width // 336) * ceil(height // 336), max_num)

class VisionQnA(VisionQnABase):
    model_name: str = "internlm-xcomposer2-4khd-7b"
    format: str = "chatml"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)
        
        self.max_tiles = extra_params.get('max_tiles', MAX_TILES)

        torch.set_grad_enabled(False)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).cuda().eval()

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        prompt, history, images, system = await prompt_history_images_system_from_messages(request.messages, img_tok = "<ImageHere>", url_handler = url_to_file)
        
        default_params = {
            'num_beams': 3,
            'do_sample': False,
            'meta_instruction': system if system else ('You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
        '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.'),
        }

        params = self.get_generation_params(request, default_params)

        with torch.cuda.amp.autocast():
            response, history = self.model.chat(self.tokenizer, query=prompt, image=images[-1], hd_num=calc_hd(images[-1], max_num=self.max_tiles), history=history, **params)

        return response
