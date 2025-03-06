from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

# qresearch/llama-3-vision-alpha-hf

# Doesn't support generation without images

class VisionQnA(VisionQnABase):
    model_name: str = "llamavision"
    format: str = "llama3"
    vision_layers: List[str] = ["mm_projector", "vision_model"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, None, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = '<image>\n' + prompt

        input_ids = self.model.tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt").unsqueeze(0).to(self.device)
        image_inputs = self.model.processor(
            images=images,
            return_tensors="pt",
            do_resize=True,
            size={"height": 384, "width": 384},
        )

        image_inputs = image_inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        image_forward_outs = self.model.vision_model(image_inputs, output_hidden_states=True)
        image_features = image_forward_outs.hidden_states[-2]
        projected_embeddings = self.model.mm_projector(image_features).to(self.device)
        embedding_layer = self.model.text_model.get_input_embeddings() # .to(self.device)
        new_embeds, attn_mask = self.model.process_tensors(input_ids, projected_embeddings, embedding_layer)

        default_params = dict(
            temperature=0.2,
            do_sample=True,
        )

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            inputs_embeds=new_embeds.to(self.device),
            attention_mask=attn_mask.to(self.device),
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            pad_token_id=self.tokenizer.eos_token_id,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.text_model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break