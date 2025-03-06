from transformers import AutoProcessor, AutoModelForImageTextToText

from vision_qna import *

class VisionQnA(VisionQnABase):
    model_name: str = "aya_vision"
    format: str = "internal"
    visual_layers: List[str] = ["multi_modal_projector", "vision_tower"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if self.params['torch_dtype'] == torch.bfloat16:
            self.dtype = self.params['torch_dtype'] = torch.float16
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(**self.params).eval() # model_id, device_map="auto", 

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()


    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        messages = messages_from_messages(request.messages)

        inputs = self.processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)

        default_params = {
            'do_sample': True,
            'temperature': 0.3,
#            'eos_token_id': self.processor.tokenizer.eos_token_id,
#            'pad_token_id': self.processor.tokenizer.eos_token_id,
        }

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.processor.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.processor.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
