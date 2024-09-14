from transformers import AutoModelForCausalLM

from vision_qna import *

# AIDC-AI/Ovis1.5-Gemma2-9B
# AIDC-AI/Ovis1.5-Llama3-8B

class VisionQnA(VisionQnABase):
    model_name: str = "generic"
    format: str = "gemma" # or llama3
    visual_layers: List[str] = ['visual_tokenizer', 'vte']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        self.params['multimodal_max_length'] = 8192

        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.conversation_formatter = self.model.get_conversation_formatter()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = "<image>\n" + prompt

        tok_chunks = [self.text_tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split('<image>')]


        input_ids = join_int_lists(tok_chunks, -200)  # -200 == <image>
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device=self.model.device)
        
        text_attention_masks = torch.ne(input_ids, self.text_tokenizer.pad_token_id).to(device=self.model.device)
        pixel_values = [ self.visual_tokenizer.preprocess_image(image).to(
            dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device) for image in images ]
        

        # hack section, skip model.generate because of cache bug with gemma2
        _, inputs_embeds, _, attention_mask = self.model.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=text_attention_masks,
            text_labels=None,
            pixel_values=pixel_values,
        )

        """
        # Hybrid cache implementation for Gemma2 - this is disabled for now, due to an error with this version of transformers
        # AttributeError: 'HybridCache' object has no attribute 'max_batch_size'
        if getattr(self.generation_config, 'cache_implementation') == 'hybrid':  # mainly for Gemma2
            kwargs['past_key_values'] = self._get_hybrid_cache_for_llm(
                getattr(kwargs, "num_beams", 1), kwargs['max_new_tokens'] + inputs_embeds.shape[-2])
            self.get_llm()._supports_cache_class = True
            kwargs['cache_implementation'] = None
        """

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
            # inputs=input_ids
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.llm.generate, tokenizer=self.text_tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.text_tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
