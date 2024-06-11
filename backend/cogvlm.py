from transformers import LlamaTokenizer, AutoModelForCausalLM

from vision_qna import *

# THUDM/cogvlm-chat-hf
# THUDM/cogagent-chat-hf
import transformers
transformers.logging.set_verbosity_error()

class VisionQnA(VisionQnABase):
    model_name: str = "cogvlm"
    format: str = 'llama2'
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
    
        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        
        query, history, images, system_message = await prompt_history_images_system_from_messages(
            request.messages, img_tok='', url_handler=url_to_image)
        
        if len(images) < 1:
            images = [ await url_to_image(transparent_pixel_url) ]

        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=history, images=images)
        
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[input_by_model['images'][0].to(self.model.device).to(self.model.dtype)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.model.device).to(self.model.dtype)]]

        params = self.get_generation_params(request)

        del params['top_k']

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
