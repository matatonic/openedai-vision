import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from loguru import logger
from vision_qna import *

# BAAI/Emu2-Chat

class VisionQnA(VisionQnABase):
    model_name: str = 'emu'
    format: str = 'emu'
    vision_layers: List[str] = ["visual", "project_up", "project_down"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False):
            if self.params['torch_dtype'] == torch.bfloat16:
                self.dtype = self.params['torch_dtype'] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(**self.params)
        else:
            checkpoint = snapshot_download(model_id)
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_pretrained(**self.params)

            max_memory=extra_params.get('max_memory', None)

            device_map = infer_auto_device_map(self.model, max_memory=max_memory, no_split_module_classes=['Block','LlamaDecoderLayer'])
            # input and output logits should be on same device
            device_map["model.decoder.lm.lm_head"] = 0

            self.model = load_checkpoint_and_dispatch(self.model, checkpoint=checkpoint, device_map=device_map).eval()

        # self.model.device/dtype are overloaded with some other object
        logger.info(f"Loaded {model_id} on device: {self.device} with dtype: {self.params['torch_dtype']}")
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt, system = await emu_images_prompt_system_from_messages(request.messages)

        if not system:
            system = "You are a helpful assistant, dedicated to delivering comprehensive and meticulous responses."

        prompt = system + prompt

        inputs = self.model.build_input_ids(
            text=[prompt],
            tokenizer=self.tokenizer,
            image=images if images else None
        )

        default_params = {
            'length_penalty': 1.0,
            'num_beams': 1, # for streaming
            'do_sample': True,
        }

        params = self.get_generation_params(request, default_params)

        generation_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image=inputs["image"].to(self.params['torch_dtype']) if images else None,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
