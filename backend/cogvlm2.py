from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

# THUDM/cogvlm2-llama3-chat-19B
# THUDM/cogvlm2-llama3-chinese-chat-19B
import transformers
transformers.logging.set_verbosity_error()

class VisionQnA(VisionQnABase):
    model_name: str = "cogvlm2"
    format: str = 'llama3'
    vision_layers: List[str] = ['vision']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
    
        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        
        query, history, images, system_message = await prompt_history_images_system_from_messages(
            request.messages, img_tok='', url_handler=url_to_image)

        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=history, images=images, template_version='chat')
        
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[input_by_model['images'][0].to(self.model.device).to(self.model.dtype)]] if images else None,
        }

        default_params = {
            'max_new_tokens': 2048,
            'pad_token_id': 128002,
            'top_p': None, # 0.9
            'temperature': None, # 0.6
        }

        params = self.get_generation_params(request, default_params)

#        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=False, skip_prompt=True)

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
