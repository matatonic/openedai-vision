from PIL import Image
from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF

from vision_qna import *

# https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two

# fancyfeast/joy-caption-alpha-two

LATEST="cgrkzexw-599808/"
LATEST_NAME="JoyCaption Alpha Two (2024-09-26a)"
CLIP_PATH = "google/siglip-so400m-patch14-384"

# This is the expected conversation format, but others work too.
SYSTEM_MSG="You are a helpful image captioner."
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}
extra_options=[
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image."
]

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # <|image_start|>, IMAGE, <|image_end|>
        other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)


class VisionQnA(VisionQnABase):
    model_name: str = "joy-caption-alpha-two"
    format: str = "llama3"
    visual_layers: List[str] = []
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
        self.clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model

        CHECKPOINT_PATH = Path(snapshot_download(repo_id="fancyfeast/joy-caption-alpha-two", repo_type="space", allow_patterns=LATEST)) / LATEST
        checkpoint = torch.load(CHECKPOINT_PATH / "clip_model.pt", map_location='cpu', weights_only=True)
        checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
        self.clip_model.load_state_dict(checkpoint)
        del checkpoint

        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH / "text_model", use_fast=True)

        del self.params['pretrained_model_name_or_path']
        self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH / "text_model", **self.params)
        self.model.eval()

        self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.model.config.hidden_size, False, False, 38, False)
        self.image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu", weights_only=True))
        self.image_adapter.eval()
        self.image_adapter.to(self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        if request.messages[0].role != 'system':
            request.messages = [Message(role='system', content=[Content(type='text', text=SYSTEM_MSG)])] + request.messages
        images, prompt = await prompt_from_messages(request.messages, self.format)

        def to_pixel_values(img):
            image = img.resize((384, 384), Image.LANCZOS)
            pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            return pixel_values.to(self.device)

        def to_image_embed(img):
            pixel_values = to_pixel_values(img)

            with torch.amp.autocast_mode.autocast('cuda', enabled=True):
                vision_outputs = self.clip_model(pixel_values=pixel_values, output_hidden_states=True)
                embedded_images = self.image_adapter(vision_outputs.hidden_states)
                return embedded_images.to(device=self.device, dtype=self.model.dtype)

        if images:
            image_embeds = [ to_image_embed(img) for img in images ]
        else:
            image_embeds =  [ await url_to_image(black_pixel_url) ]

        split_tokens = [ self.tokenizer.encode(c, return_tensors="pt", add_special_tokens=False, truncation=False) for c in prompt.split('<image>') ]
        split_embeds = [ self.model.model.embed_tokens(tok_ids.to(self.device)) for tok_ids in split_tokens ]

        input_ids = [split_tokens[0]]
        inputs_embeds = [ split_embeds[0] ]

        for im, tok, emb in zip(image_embeds, split_tokens[1:], split_embeds[1:]):
            input_ids.extend([torch.zeros((1, im.shape[1]), dtype=torch.long), tok])
            inputs_embeds.extend([ im, emb ])

        input_ids = torch.cat(input_ids, dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        inputs_embeds = torch.cat(inputs_embeds, dim=1).to(self.device)
        
        inputs = dict(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        default_params = dict(
            max_new_tokens=512,
            do_sample=True,
            suppress_tokens=None,
        )

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
