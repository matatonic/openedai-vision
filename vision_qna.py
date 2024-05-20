import io
import uuid
import requests
from datauri import DataURI
from PIL import Image
import torch
from typing import Optional, List, Literal
from pydantic import BaseModel
from transformers import BitsAndBytesConfig
from transformers.image_utils import load_image

class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto" # auto -> low (512) or high (Nx512) based on res.

class Content(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

class Message(BaseModel):
    role: str
    content: List[Content]

class ImageChatRequest(BaseModel):
    model: str # = "gpt-4-vision-preview"
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = None
    top_p: float = None

class VisionQnABase:
    model_name: str = None
    format: str = None
    revision: str = 'main'
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        self.device, self.dtype = self.select_device_dtype(device)

        self.params = {
            'pretrained_model_name_or_path': model_id,
            'torch_dtype': self.dtype,
            'low_cpu_mem_usage': True,
            'revision': self.revision,
            'device_map': device_map,
        }
        if extra_params.get('load_in_4bit', False):
            load_in_4bit_params = {
                'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True, # XXX make this an option
                    bnb_4bit_compute_dtype=self.dtype,
                    load_in_4bit_fp32_cpu_offload=False,
                )
            }
            self.params.update(load_in_4bit_params)
        elif extra_params.get('load_in_8bit', False):
            load_in_8bit_params = {
                'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,
                    load_in_8bit_fp32_cpu_offload=False,
                )
            }
            self.params.update(load_in_8bit_params)

        if extra_params.get('use_flash_attn', False):
            flash_attn_params = {
                "attn_implementation": "flash_attention_2",
            }
            self.params.update(flash_attn_params)

        if extra_params.get('trust_remote_code', False):
            flash_attn_params = {
                "trust_remote_code": True,
            }
            self.params.update(flash_attn_params)

        if format:
            self.format =  format


    def select_device(self):
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    def select_dtype(self, device):
        return torch.float32 if device == 'cpu' else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def select_device_dtype(self, device):
        device = self.select_device() if device == 'auto' else device
        dtype = self.select_dtype(device)
        return device, dtype
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        pass

    def get_generation_params(self, request: ImageChatRequest, default_params = {}) -> dict:
        params = {
            'top_k': None,
            'do_sample': False,
            'use_cache': True,
        }
        params.update(default_params)

        if request.max_tokens:
            params["max_new_tokens"] = request.max_tokens
        
        if request.temperature is not None:
            if request.temperature > 0:
                params["do_sample"] = True
                params["temperature"] = request.temperature

        if request.top_p is not None and request.top_p != params.get('top_p', 1.0):
            params["do_sample"] = True
            params["top_p"] = request.top_p

        return params

async def url_to_image(img_url: str) -> Image.Image:
    #return load_image(img_url)
    if img_url.startswith('http'):
        response = requests.get(img_url)

        img_data = response.content
    elif img_url.startswith('data:'):
        img_data = DataURI(img_url).data

    return Image.open(io.BytesIO(img_data)).convert("RGB")

async def url_to_file(img_url: str) -> str:
    if img_url.startswith('data:'):
        # secure temp filename
        filename = f"/tmp/{uuid.uuid4()}"
        with open(filename, 'wb') as f:
            f.write(DataURI(img_url).data)
            return filename
    else:
        response = requests.get(img_url)
        filename = f"/tmp/{uuid.uuid4()}"
        with open(filename, 'wb') as f:
            f.write(response.content)
            return filename

async def images_hfmessages_from_messages(messages: list[Message], url_handler = url_to_image):
    hfmessages = []
    images = []

    for m in messages:
        content = []
        for c in m.content:
            if c.type == 'image_url':
                image = await url_handler(c.image_url.url)
                images.extend([image])
                content.extend([{"type": "image"}])
            elif c.type == 'text':
                content.extend([{'type': 'text', 'text': c.text}])

        hfmessages.extend([{'role': m.role, 'content': content}])

    return images, hfmessages


async def phi15_prompt_from_messages(messages: list[Message], img_tok = "<image>", img_end = ''): # </image>
    prompt = ''
    images = []

    for m in messages:
        if m.role == 'user':
            p = ''
            for c in m.content:
                if c.type == 'image_url':
                    images.extend([ await url_to_image(c.image_url.url) ])
                    p = img_tok + p + img_end
                if c.type == 'text':
                    p += f"{c.text}\n\n" # Question:
            prompt += p
        elif m.role == 'assistant':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"Answer: {c.text}\n\n"
        elif m.role == 'system':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"{c.text}\n\n"  # fake system prompt

    prompt += "Answer:"

    return images, prompt

async def vicuna0_prompt_from_messages(messages: list[Message], img_tok = "<image_placeholder>\n"):
    prompt = ''
    images = []

    for m in messages:
        if m.role == 'user':
            text = ''
            has_image = False

            for c in m.content:
                if c.type == 'image_url':
                    images.extend([ await url_to_image(c.image_url.url) ])
                    has_image = True
                if c.type == 'text':
                    text = c.text
            
            img_tag = img_tok if has_image else ''
            prompt += f"### Human: {img_tag}{text}\n"
        elif m.role == 'assistant':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"### Assistant: {c.text}\n"
        elif m.role == 'system':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"{c.text}\n\n"

    prompt += "### Assistant:"

    return images, prompt

async def vicuna_prompt_from_messages(messages: list[Message], img_tok = "<image>\n"):
    prompt = ''
    images = []

    for m in messages:
        if m.role == 'user':
            text = ''
            has_image = False

            for c in m.content:
                if c.type == 'image_url':
                    images.extend([ await url_to_image(c.image_url.url) ])
                    has_image = True
                if c.type == 'text':
                    text = c.text
            
            img_tag = img_tok if has_image else ''
            prompt += f"USER: {img_tag}{text}\n"
        elif m.role == 'assistant':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"ASSISTANT: {c.text}\n"
        elif m.role == 'system':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"{c.text}\n\n"

    prompt += "ASSISTANT:"

    return images, prompt

async def llama2_prompt_from_messages(messages: list[Message], img_tok = "<image>\n"):
    prompt = ''
    images = []

    for m in messages:
        if m.role == 'user':
            text = ''
            has_image = False

            for c in m.content:
                if c.type == 'image_url':
                    images.extend([ await url_to_image(c.image_url.url) ])
                    has_image = True
                if c.type == 'text':
                    text = c.text

            img_tag = img_tok if has_image else ''
            prompt += f"[INST] {img_tag}{text} [/INST]"
        elif m.role == 'assistant':
            for c in m.content:
                if c.type == 'text':
                    prompt += f" {c.text}"
        elif m.role == 'system':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"[INST] <<SYS>>\n{c.text}\n<</SYS>> [/INST]" # not quite right, but it's a start

    return images, prompt

async def llama3_prompt_from_messages(messages: list[Message], img_tok = "<image>"):
    prompt = ''
    images = []

    for m in messages:
        has_image = False
        
        for c in m.content:
            if c.type == 'image_url':
                images.extend([ await url_to_image(c.image_url.url) ])
                has_image = True
        
        img_tag = img_tok if has_image else ''

        for c in m.content:
            if c.type == 'text':
                prompt += f"<|start_header_id|>{m.role}<|end_header_id|>\n\n{img_tag}{c.text.strip()}<|eot_id|>"
    
    prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'

    return images, prompt

async def chatml_prompt_from_messages(messages: list[Message], img_tok = "<image>\n"):
    prompt = ''
    images = []

    for m in messages:
        if m.role == 'user':
            text = ''
            has_image = False

            for c in m.content:
                if c.type == 'image_url':
                    images.extend([ await url_to_image(c.image_url.url) ])
                    has_image = True
                if c.type == 'text':
                    text = c.text

            img_tag = img_tok if has_image else ''
            prompt += f"<|im_start|>user\n{img_tag}{text}<|im_end|>"
        elif m.role == 'assistant':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"<|im_start|>assistant\n{c.text}<|im_end|>"
        elif m.role == 'system':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"<|im_start|>system\n{c.text}<|im_end|>"

    prompt += f"<|im_start|>assistant\n"

    return images, prompt

async def gemma_prompt_from_messages(messages: list[Message], img_tok = "<image>\n"):
    prompt = ''
    images = []

    for m in messages:
        if m.role == 'user':
            text = ''
            has_image = False

            for c in m.content:
                if c.type == 'image_url':
                    images.extend([ await url_to_image(c.image_url.url) ])
                    has_image = True
                if c.type == 'text':
                    text = c.text

            img_tag = img_tok if has_image else ''
            prompt += f"<start_of_turn>user\n{img_tag}{text}<end_of_turn>"
        elif m.role == 'assistant':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"<start_of_turn>model\n{c.text}<end_of_turn>"
        elif m.role == 'system':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"<start_of_turn>system\n{c.text}<end_of_turn>" # fake it


    prompt += f"<start_of_turn>model\n"

    return images, prompt

async def fuyu_prompt_from_messages(messages: list[Message], img_tok = "", img_end = ''):
    prompt = ''
    images = []

    for m in messages:
        if m.role == 'user':
            p = ''
            for c in m.content:
                if c.type == 'image_url':
                    images.extend([ await url_to_image(c.image_url.url) ])
                    p = img_tok + p + img_end
                if c.type == 'text':
                    p += f"{c.text}\n\n" # Question:
            prompt += p
        elif m.role == 'assistant':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"\x04{c.text}\n"
        elif m.role == 'system':
            for c in m.content:
                if c.type == 'text':
                    prompt += f"{c.text}\n\n" # fake system prompt doesn't work.

    return images, prompt

async def emu_images_prompt_system_from_messages(messages: list[Message], img_tok = "[<IMG_PLH>]"):
    prompt = ''
    images = []
    system_message = None

    for m in messages:
        if m.role == 'user':
            text = ''
            has_image = False

            for c in m.content:
                if c.type == 'image_url':
                    images.extend([ await url_to_image(c.image_url.url) ])
                    has_image = True
                if c.type == 'text':
                    text = c.text
            
            img_tag = img_tok if has_image else ''
            prompt += f" [USER]: {img_tag}{text}"
        elif m.role == 'assistant':
            for c in m.content:
                if c.type == 'text':
                    prompt += f" [ASSISTANT]: {c.text}</s>"
        elif m.role == 'system':
            for c in m.content:
                if c.type == 'text':
                    system_message = c.text

    prompt += " [ASSISTANT]:"

    return images, prompt, system_message

async def prompt_history_images_system_from_messages(messages: list[Message], img_tok = "<image>\n", url_handler = url_to_image):
    history = []
    images = []
    prompt = ''
    system_prompt = None

    for m in messages:
        if m.role == 'user':
            p = ''
            for c in m.content:
                if c.type == 'image_url':
                    image = await url_handler(c.image_url.url)
                    images.extend([image])
                    p = img_tok + p
                if c.type == 'text':
                    p += c.text

            prompt += p
        elif m.role == 'assistant':
            for c in m.content:
                if c.type == 'text':
                    history.extend([(prompt, c.text)])
                    prompt = ''
        elif m.role == 'system':
            for c in m.content:
                if c.type == 'text':
                    system_prompt = c.text

    return prompt, history, images, system_prompt




async def prompt_from_messages(messages: list[Message], format: str) -> str:
    known_formats = {
        'phi15': phi15_prompt_from_messages,
        'vicuna0': vicuna0_prompt_from_messages,
        'vicuna': vicuna_prompt_from_messages,
        'llama2': llama2_prompt_from_messages,
        'llama3': llama3_prompt_from_messages,
        'mistral': llama2_prompt_from_messages, # simplicity
        'chatml': chatml_prompt_from_messages,
        'gemma': gemma_prompt_from_messages,
        'fuyu': fuyu_prompt_from_messages,
    }

    if format not in known_formats:
        raise ValueError(f"Unknown format: {format}")
    
    return await known_formats[format](messages)

def guess_model_format(model_name: str) -> str:
    model_id = model_name.lower()

    model_format_match_map = {
        'llama2': ['bakllava', '8x7b', 'mistral', 'mixtral'],
        'llama3': ['llama-3-vision', '360vl'],
        'gemma': ['gemma', '-2b'],
        'vicuna': ['vicuna', '13b'],
        'vicuna0': ['yi-vl'],
        'phi15': ['moondream1', 'moondream2', 'monkey'],
        'chatml': ['34b', 'yi-6b', 'nanollava', 'internvl-chat-v1-5'],
        'fuyu': ['fuyu'],
    }
    for format, options in model_format_match_map.items():
        if any(x in model_id for x in options):
            return format

    return 'vicuna'

def guess_backend(model_name: str) -> str:
    model_id = model_name.lower()

    if 'nanollava' in model_id:
        return 'nanollava'

    if 'llava' in model_id:
        if 'v1.6' in model_id:
            return 'llavanext'
        return 'llava'

    if 'qwen' in model_id:
        return 'qwen-vl'
    
    if 'moondream1' in model_id:
        return 'moondream1'
    
    if 'moondream2' in model_id:
        return 'moondream2'

    if 'monkey' in model_id:
        return 'monkey'
    
    if 'mgm-' in model_id or 'minigemini' in model_id or 'mini-gemini' in model_id:
        return 'minigemini'

    if 'deepseek' in model_id:
        return 'deepseek-vl'

    if 'minicpm-v' in model_id:
        return 'omnilmm3b'

    if 'omnilmm-12b' in model_id:
        return 'omnilmm12b'

    if 'xcomposer2-4khd' in model_id:
        return 'xcomposer2-4khd'
        
    if 'xcomposer2-vl' in model_id:
        return 'xcomposer2-vl'
    
    if 'xcomposer2' in model_id:
        return 'xcomposer2'

    if 'yi-vl' in model_id:
        return 'yi-vl'

    if 'cogvlm2' in model_id:
        return 'cogvlm2'

    if 'cogagent-' in model_id or 'cogvlm-' in model_id:
        return 'cogvlm'
    
    if 'fuyu' in model_id:
        return 'fuyu'
    
    if 'internvl-chat-v1-5' in model_id:
        return 'internvl-chat-v1-5'
    
    if 'idefics2' in model_id:
        return 'idefics2'
    
    if 'llama-3-vision-alpha' in model_id:
        return 'llama3vision'
    
    if 'bunny' in model_id:
        return 'bunny'
    
    if 'mantis' in model_id:
        return 'mantis'
    
    if 'emu' in model_id:
        return 'emu'
    
    if '360vl' in model_id:
        return '360vl'