from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
import torch.amp.autocast_mode
from huggingface_hub import hf_hub_download

from vision_qna import *

# fancyfeast/joy-caption-pre-alpha

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class VisionQnA(VisionQnABase):
    model_name: str = "joy-caption-pre-alpha"
    format: str = "llama3"
    visual_layers: List[str] = []
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):

        logger.warning("Loading fancyfeast/joy-caption-pre-alpha with wpkklhc6/image_adapter.pt, ")
        # XXX Ignore the actual model_id
        if extra_params.get("load_in_4bit", False):
            model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit" # no authorization required
        else:
            model_id = "meta-llama/Meta-Llama-3.1-8B" # requires authorized access

        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        CLIP_PATH = "google/siglip-so400m-patch14-384"

        self.clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
        self.clip_model = AutoModel.from_pretrained(CLIP_PATH)
        self.clip_model = self.clip_model.vision_model
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model.to(self.device)

        self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.model.config.hidden_size)
        CHECKPOINT_PATH = hf_hub_download(repo_id="fancyfeast/joy-caption-pre-alpha", repo_type="space", subfolder="wpkklhc6", filename="image_adapter.pt")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        self.image_adapter.load_state_dict(checkpoint)
        self.image_adapter.eval()
        self.image_adapter.to(self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        #prompt = "A descriptive caption for this image:\n"

        # Tokenize the prompt
        prompt_tok = self.tokenizer.encode(prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)

        if len(images) < 1:
            inputs = dict(
                input_ids=prompt_tok.to(device=self.device),
            )
        else:
            # Preprocess image
            image = self.clip_processor(images=images[0], return_tensors='pt').pixel_values.to(device=self.device)

            # Embed image
            with torch.amp.autocast_mode.autocast('cuda', enabled=True):
                vision_outputs = self.clip_model(pixel_values=image, output_hidden_states=True)
                image_features = vision_outputs.hidden_states[-2]
                embedded_images = self.image_adapter(image_features)
                embedded_images = embedded_images.to(self.device)
        
            # Embed prompt
            prompt_embeds = self.model.model.embed_tokens(prompt_tok.to(device=self.device))
            embedded_bos = self.model.model.embed_tokens(torch.tensor([[self.tokenizer.bos_token_id]], device=self.device, dtype=torch.int64))

            # Construct prompts
            inputs_embeds = torch.cat([
                embedded_bos.expand(embedded_images.shape[0], -1, -1),
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(embedded_images.shape[0], -1, -1),
            ], dim=1)

            input_ids = torch.cat([
                torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                prompt_tok,
            ], dim=1).to(device=self.device)
            attention_mask = torch.ones_like(input_ids)

            inputs = dict(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        default_params = dict(
            max_new_tokens=300,
            do_sample=True,
            top_k=10,
            temperature=0.5,
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
