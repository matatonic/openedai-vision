from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

import transformers
import warnings
# disable some warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# qnguyen3/nanoLLaVA
# qnguyen3/nanoLLaVA-1.5

def join_int_lists(int_lists, separator):
    result = []
    for i, lst in enumerate(int_lists):
        result.extend(lst)
        if i < len(int_lists) - 1:
            result.append(separator)
    return result

class VisionQnA(VisionQnABase):
    model_name: str = "nanollava"
    format: str = "chatml"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)
        
        torch.set_default_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
            self.model = self.model.to(self.device)

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        text_chunks = [self.tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
        if images:
            text_with_img_tok = join_int_lists(text_chunks, -200) # -200 == <image>
            encoded_images = self.model.process_images(images, self.model.config).to(dtype=self.model.dtype)
        else:
            text_with_img_tok = text_chunks[0]
            encoded_images = None

        input_ids = torch.tensor(text_with_img_tok, dtype=torch.long).unsqueeze(0)

        default_params = {
            'top_p': 0.8,
            'temperature': 0.7,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            input_ids=input_ids,
            images=encoded_images,
            **params
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
