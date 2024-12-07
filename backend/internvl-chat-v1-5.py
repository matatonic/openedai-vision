import os
from threading import Thread
from transformers import AutoTokenizer, AutoModel
from vision_qna import *
import math
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# OpenGVLab/InternVL-Chat-V1-5
# OpenGVLab/InternVL-Chat-V1-5-Int8
# OpenGVLab/Mini-InternVL-Chat-2B-V1-5
# OpenGVLab/Mini-InternVL-Chat-4B-V1-5 (phintern)
# OpenGVLab/InternVL2-1B
# OpenGVLab/InternVL2-2B-AWQ (empty response)
# OpenGVLab/InternVL2-2B
# OpenGVLab/InternVL2-4B
# OpenGVLab/InternVL2-4B (phintern)
# OpenGVLab/InternVL2-8B
# OpenGVLab/InternVL2-26B
# OpenGVLab/InternVL2-40B (yi-34- nous-hermes-2)
# OpenGVLab/InternVL2-Llama3-76B
# OpenGVLab/InternVL2_5-1B
# OpenGVLab/InternVL2_5-2B
# OpenGVLab/InternVL2_5-4B
# OpenGVLab/InternVL2_5-8B
# OpenGVLab/InternVL2_5-26B
# OpenGVLab/InternVL2_5-38B
# OpenGVLab/InternVL2_5-78B


MAX_TILES = 6

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=MAX_TILES, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=MAX_TILES):
    #image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'Mini-InternVL-2B-V1-5': 24, 'Mini-InternVL-4B-V1-5': 32, 'InternVL-Chat-V1-5': 48,
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80,
        'InternVL2_5-1B': 24, 'InternVL_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def split_model_dynamic(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'Mini-InternVL-2B-V1-5': 24, 'Mini-InternVL-4B-V1-5': 32, 'InternVL-Chat-V1-5': 48,
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80,
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}.get(model_name, None)
    if num_layers is None:
        logger.warning(f"Unknown model name {model_name}, can't guess layers count, reverting to 'auto' device_map which may not work.")
        return 'auto'
    # Get the available memory on each GPU
    gpu_memory = [torch.cuda.get_device_properties(i).total_memory for i in range(world_size)]
    # Subtract half of the first GPU's memory, XXX try to do better
    reserved = 16 if num_layers > 32 else 6 # internvit 6B vs 300M, unfinished hack.

    gpu_memory[0] -= min(reserved * 1e9, gpu_memory[0])
    # Calculate the total available memory
    total_memory = sum(gpu_memory)
    # Calculate the memory ratios
    memory_ratios = [mem / total_memory for mem in gpu_memory]
    # Assign layers to GPUs based on their memory ratios
    layer_cnt = 0
    for i, ratio in enumerate(memory_ratios):
        num_layers_on_gpu = math.floor(num_layers * ratio)
        for j in range(num_layers_on_gpu):
            if layer_cnt < num_layers:
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
    # Assign any remaining layers to the first GPU
    while layer_cnt < num_layers:
        device_map[f'language_model.model.layers.{layer_cnt}'] = 0
        layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    #device_map['language_model'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    logger.debug(f"{device_map=}")

    return device_map

class VisionQnA(VisionQnABase):
    model_name: str = "internvl-chat-v1-5"
    format: str = "chatml"
    vision_layers: List[str] = ["vision_model"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):

        # This doesn't work at all for me, still complains
        # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
        #if device_map == 'auto':
        #    model_name = model_id.split('/')[-1]
        #    device_map = split_model_dynamic(model_name)

        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        self.max_tiles = extra_params.get('max_tiles', MAX_TILES)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        for name, module in self.model.named_modules():
            # Check the device of the module's parameters
            dev = 'UNKNOWN'
            if hasattr(module, 'weight'):
                dev = module.weight.device
            elif hasattr(module, 'bias'):
                dev = module.bias.device
            elif hasattr(module, 'device'):
                dev = module.device
            logger.debug(f'Layer: {name}, Device: {dev}')

        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

        self.eos_token = '<|end|>' if self.format == 'phintern' else '<|im_end|>'

        if self.tokenizer.convert_tokens_to_ids(self.eos_token) != 0:
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)  # 92542, InternLM2
        else:
            self.eos_token_id = self.tokenizer.eos_token_id

        self.loaded_banner()
    

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)
        
        # TODO: use detail to set max tiles if detail=low (=512)
        # if .detail == 'low': max_num=1
        images = [load_image(image, max_num=self.max_tiles).to(self.model.dtype).cuda() for image in images]
        if len(images) > 1:
            pixel_values = torch.cat(images, dim=0)
        elif len(images) > 0:
            pixel_values = images[0]
        else:
            pixel_values = None
        
        if pixel_values is not None:
            for img in images:
                image_tokens = '<img>' + '<IMG_CONTEXT>' * self.model.num_image_token * img.size(0) + '</img>'
                prompt = prompt.replace('<image>', image_tokens, 1)
        
        model_inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()

        default_params = {
            'num_beams': 1,
            'max_new_tokens': 512,
            'do_sample': False,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.eos_token_id,
        }

        params = self.get_generation_params(request, default_params)

        del params['use_cache']
        
        generation_kwargs = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
