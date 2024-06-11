from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria

from vision_qna import *

# Salesforce/xgen-mm-phi3-mini-instruct-r-v1

class VisionQnA(VisionQnABase):
    model_name: str = "generic"
    format: str = 'phi3'
    vision_layers: List[str] = ['vision_encoder', 'vision_tokenizer']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        # Doesn't work with accelerate
        # Errors:
        # NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
        # also doesn't work with --load-in-4bit for the same reason
        self.params['low_cpu_mem_usage'] = False
        del self.params['device_map']

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False), use_fast=False, legacy=False)
        self.model = AutoModelForVision2Seq.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

        self.image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.eos_token = "<|end|>"

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await phi3_prompt_from_messages(request.messages, img_tok = "<image>\n")
        default_system = ("A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.")

        language_inputs = self.tokenizer([prompt], return_tensors="pt")
        if images:
            inputs = self.image_processor(images, return_tensors="pt", image_aspect_ratio='anyres').to(dtype=self.model.dtype)
            inputs.update(language_inputs)
            inputs = {name: tensor.to(device=self.model.device) for name, tensor in inputs.items()}
        else:
            inputs = language_inputs.to(device=self.model.device)
            inputs['pixel_values'] = None

        default_params = {
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': 32007, # <|end|>
            'do_sample': False,
            'max_new_tokens': 768,
            'top_p': None,
            'num_beams': 1,
            'image_size': [img.size for img in images],
        }

        params = self.get_generation_params(request, default_params=default_params)

        # errors
        del params['use_cache']

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
