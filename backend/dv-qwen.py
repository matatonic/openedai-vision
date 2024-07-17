from transformers import AutoTokenizer, AutoModelForCausalLM, logging

import warnings

from vision_qna import *

# disable some warnings
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# cognitivecomputations/dolphin-vision-72b
# cognitivecomputations/dolphin-vision-7b

class VisionQnA(VisionQnABase):
    model_name: str = "dolphin-vision"
    format: str = "chatml"
    visual_layers: List[str] = ["vision_tower", "mm_projector"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        if not images:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            image_tensor = None
        else:
            text_chunks = [self.tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
            input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(self.model.device)

            image_tensor = self.model.process_images(images, self.model.config).to(dtype=self.model.dtype, device=self.model.device)

        params = self.get_generation_params(request)

        generation_kwargs = dict(
            input_ids=input_ids,
            images=image_tensor,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
