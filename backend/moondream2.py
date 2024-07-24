import re
from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

# vikhyatk/moondream2

class VisionQnA(VisionQnABase):
    model_name: str = "moondream2"
    revision: str = '2024-07-23' # 'main'
    format: str = 'phi15'
    vision_layers: List[str] = ["vision_encoder"]

    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        # not supported yet
        del self.params['device_map']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
    
        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        encoded_images = self.model.encode_image(images) if images else None
        inputs_embeds = self.model.input_embeds(prompt, encoded_images, self.tokenizer)

        params = self.get_generation_params(request)
        
        #streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=False, skip_prompt=True)

        generation_kwargs = dict(
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.bos_token_id,
            inputs_embeds=inputs_embeds,
            **params,
        )
        for new_text in threaded_streaming_generator(generate=self.model.text_model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
