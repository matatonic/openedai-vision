from transformers import AutoModelForCausalLM

from vision_qna import *

#AIDC-AI/Ovis2-1B
#AIDC-AI/Ovis2-2B
#AIDC-AI/Ovis2-4B
#AIDC-AI/Ovis2-8B
#AIDC-AI/Ovis2-16B
#AIDC-AI/Ovis2-34B

IMAGE_TOKEN = "<image>"

class VisionQnA(VisionQnABase):
    model_name: str = "ovis2"
    format: str = "custom"
    visual_layers: List[str] = ['visual_tokenizer', 'vte']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        conversation = []
        images = []

        for m in request.messages:
            content = ''
            for c in m.content:
                if c.type == 'image_url':
                    image = await url_to_image(c.image_url.url)
                    images.extend([image])
                    content += IMAGE_TOKEN + '\n'
                elif c.type == 'text':
                    content += c.text

            if content:
                if m.role == 'user':
                    conversation.extend([{'from': 'human', 'value': content }])
                elif m.role == 'assistant':
                    conversation.extend([{'from': 'gpt', 'value': content }])
                # system is ignored

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            conversation[0]['value'] = IMAGE_TOKEN + '\n' + conversation[0]['value']

        _prompt, input_ids, pixel_values = self.model.preprocess_inputs(conversation, images)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        default_params =  dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True,
            num_beams=1,
        )

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            inputs=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.text_tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.text_tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
