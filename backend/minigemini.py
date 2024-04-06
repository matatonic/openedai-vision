import re
from transformers import AutoTokenizer, AutoModelForCausalLM

from minigemini.model.builder import load_pretrained_model
from minigemini.mm_utils import process_images

from vision_qna import *

class VisionQnA(VisionQnABase):
    model_name: str = "minigemini"
    format: str = "llama2"
    
    def __init__(self, model_id: str, device: str, extra_params = {}, format = None):
        super().__init__(model_id, device, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        #encoded_images = self.model.encode_image(images).to(self.device)
        # square?
        image_tensor = process_images(image_convert, image_processor, model.config)
        image_processor(images, return_tensors='pt')['pixel_values']

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        params = self.get_generation_params(request)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                images_aux=None,
                bos_token_id=self.tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=self.tokenizer.pad_token_id,  # Pad token
                use_cache=True,
                **params,
            )
            
        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        self.

        return answer



