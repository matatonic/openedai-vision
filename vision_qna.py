
import io
import requests
from datauri import DataURI
from PIL import Image
import torch

class VisionQnABase:
    model_name: str = None
    
    def __init__(self, model_id: str, device: str, extra_params = {}):
        pass

    async def url_to_image(self, img_url: str) -> Image.Image:
        if img_url.startswith('http'):
            response = requests.get(img_url)
            
            img_data = response.content
        elif img_url.startswith('data:'):
            img_data = DataURI(img_url).data

        return Image.open(io.BytesIO(img_data)).convert("RGB")

    def select_device(self):
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    async def single_question(self, image_url: str, prompt: str) -> str:
        pass
