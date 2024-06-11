import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

# echo840/Monkey
# echo840/Monkey-Chat

class VisionQnA(VisionQnABase):
    model_name: str = "monkey"
    format: str = 'phi15' # phi15-ish
    vision_layers: List[str] = ["vision", "vision_tower", "resampler", "visual", "in_proj","out_proj","c_fc","c_proj"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.eos_token = self.tokenizer.decode(self.tokenizer.eod_id) # <|endoftext|>

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        try: # 
            files, prompt = await phi15_prompt_from_messages(request.messages, img_tok = "<img>{}</img> ", img_end = '', url_handler = url_to_file)

            input_ids = self.tokenizer(prompt, return_tensors='pt', padding='longest')

            attention_mask = input_ids.attention_mask.to(self.model.device)
            input_ids = input_ids.input_ids.to(self.model.device)

            default_params = {
                'top_p': None,
                'do_sample': False,
            }

            params = self.get_generation_params(request, default_params=default_params)

            generation_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=1,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,
                **params,
            )

            for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
                end = new_text.find(self.eos_token)
                if end == -1:
                    yield new_text
                else:
                    yield new_text[:end]
                    break

        
        finally:
            for f in files:
                os.remove(f)

