#import argparse
#import os
#import sys
#from PIL import Image
#from threading import Thread
#import torch
#from transformers import TextIteratorStreamer

#import collections
#import collections.abc
#for type_name in collections.abc.__all__:
#    setattr(collections, type_name, getattr(collections.abc, type_name))
#
#from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
#from deepseek_vl.utils.io import load_pretrained_model

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"


conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image.",
        "images": ["./images/training_pipelines.png"]
    },
    {
        "role": "Assistant",
        "content": ""
    }
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)


import io
import requests
from datauri import DataURI
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class VisionQnA:
    model_name: str = "deepseek-vl"
    
    def __init__(self, model_id: str, device: str):
#        if device == 'auto':
#            device = self.select_device()

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_id)

        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        self.model = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def select_device(self):
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    async def url_to_image(self, img_url: str) -> Image.Image:
        if img_url.startswith('http'):
            response = requests.get(img_url)
            
            img_data = response.content
        elif img_url.startswith('data:'):
            img_data = DataURI(img_url).data

        return Image.open(io.BytesIO(img_data)).convert("RGB")
    
    async def single_question(self, image_url: str, prompt: str) -> str:
        image = await self.url_to_image(image_url)

        prepare_inputs = self.vl_chat_processor(conversations=prompts, images=pil_images, force_batchify=True).to(self.model.device)

        return generate(self.model, self.vl_chat_processor.tokenizer, prepare_inputs)
