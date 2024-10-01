from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from vision_qna import *

# nvidia/NVLM-D-72B

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
    model_name: str = "nvlm"
    format: str = "chatml"
    vision_layers: List[str] = ["vision_model"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.max_tiles = extra_params.get('max_tiles', MAX_TILES)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        self.eos_token = '<|im_end|>'
        self.IMG_CONTEXT_TOKEN='<|vision_pad|>'
        self.IMG_START_TOKEN = '<Image>' # <|vision_start|> ?
        self.IMG_END_TOKEN = '<Image>' # <|vision_end|> ?
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)
        
        if len(images) < 1:
            pixel_values = None
        else:
            pixel_values = load_image(images[-1], max_num=self.max_tiles).to(self.model.dtype).cuda()
        
            for num_patches in [pixel_values.shape[0]]:
                tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)] + ["<tile_global_thumbnail>"]
                image_tokens = ''
                for tile_pos_identifier in tile_pos_identifiers:
                    image_tokens += tile_pos_identifier + self.IMG_CONTEXT_TOKEN * self.model.num_image_token
                image_tokens = self.IMG_START_TOKEN + image_tokens + self.IMG_END_TOKEN
                prompt = prompt.replace('<image>', image_tokens, 1)

        model_inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()

        default_params = dict(
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

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
