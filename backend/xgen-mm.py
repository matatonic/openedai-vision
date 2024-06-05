from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria

from vision_qna import *

# Salesforce/xgen-mm-phi3-mini-instruct-r-v1

class VisionQnA(VisionQnABase):
    model_name: str = "generic"
    format: str = 'phi3'
    vision_layers: List[str] = ['vision_encoder']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        # Doesn't work with accelerate
        # Errors:
        # NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
        self.params['low_cpu_mem_usage'] = False
        del self.params['device_map']

        self.model = AutoModelForVision2Seq.from_pretrained(**self.params).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False), use_fast=False, legacy=False)
        self.image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await phi3_prompt_from_messages(request.messages, img_tok = "<image>\n")
        default_system = ("A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.")

        inputs = self.image_processor(images, return_tensors="pt", image_aspect_ratio='anyres').to(dtype=self.model.dtype)
        language_inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        inputs = {name: tensor.to(device=self.model.device) for name, tensor in inputs.items()}

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

        output = self.model.generate(**inputs, **params)

        response = self.tokenizer.decode(output[0], skip_special_tokens=True).split("<|end|>")[0]

        return response
