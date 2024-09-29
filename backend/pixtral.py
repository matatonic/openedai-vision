
from huggingface_hub import snapshot_download
from safetensors import safe_open
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from vision_qna import *

# mistralai/Pixtral-12B-2409

class VisionQnA(VisionQnABase):
    model_name: str = "pixtral"
    format: str = "pixtral"
    visual_layers: List[str] = ["vision_encoder", 'vision_language_adapter']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        mistral_models_path = snapshot_download(repo_id=model_id, allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"])

        self.tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
        self.model = Transformer.from_folder(mistral_models_path, device=self.device, dtype=self.dtype)

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        #if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
        #   self.model = self.model.to(self.device)

        self.loaded_banner()
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        prompt = await pixtral_messages(request.messages)

        # tokenize image urls and text
        tokenized = self.tokenizer.encode_chat_completion(prompt)

        generation_kwargs = dict(
            eos_id = self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
            max_tokens = request.max_tokens,
            temperature= 0.35 if request.temperature is None else request.temperature,
        )

        tps_start = time.time()
        out_tokens, _ = generate([tokenized.tokens], self.model, images=[tokenized.images], **generation_kwargs)
        logger.info(f"Generated {len(out_tokens[0])} tokens at {len(out_tokens[0]) / (time.time() - tps_start):0.2f} T/s")

        return self.tokenizer.decode(out_tokens[0])
