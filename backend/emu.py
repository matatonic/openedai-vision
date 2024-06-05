import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

from vision_qna import *

# "BAAI/Emu2-Chat"

class VisionQnA(VisionQnABase):
    model_name: str = 'emu'
    format: str = 'emu'
    vision_layers: List[str] = ["visual", "project_up", "project_down"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if self.params['torch_dtype'] == torch.bfloat16:
            self.params['torch_dtype'] = torch.float16

        checkpoint = snapshot_download(model_id)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(**self.params)

        max_memory=extra_params.get('max_memory', None)

        device_map = infer_auto_device_map(self.model, max_memory=max_memory, no_split_module_classes=['Block','LlamaDecoderLayer'])
        # input and output logits should be on same device
        device_map["model.decoder.lm.lm_head"] = 0

        self.model = load_checkpoint_and_dispatch(self.model, checkpoint=checkpoint, device_map=device_map).eval()
        """
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
        """

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # self.model.device/dtype are overloaded with some other object
        print(f"Loaded on device: {self.device} with dtype: {self.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt, system = await emu_images_prompt_system_from_messages(request.messages)

        if not system:
            system = "You are a helpful assistant, dedicated to delivering comprehensive and meticulous responses."

        prompt = system + prompt

        inputs = self.model.build_input_ids(
            text=[prompt],
            tokenizer=self.tokenizer,
            image=images
        )
        # .cuda()

        default_params = {
            'length_penalty': -1,
        }

        params = self.get_generation_params(request, default_params)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.float16), # should be torch.float16
                **params,
            )

            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return response
