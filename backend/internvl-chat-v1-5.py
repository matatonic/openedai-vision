import os
from threading import Thread
from transformers import AutoTokenizer, AutoModel
from vision_qna import *
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


class VisionQnA(VisionQnABase):
    model_name: str = "internvl-chat-v1-5"
    format: str = "chatml"
    vision_layers: List[str] = ["vision_model"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        self.max_tiles = extra_params.get('max_tiles', MAX_TILES)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

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
