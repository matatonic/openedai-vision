from transformers import AutoTokenizer, AutoModelForCausalLM

from vision_qna import *

# "qresearch/llama-3-vision-alpha-hf"

class VisionQnA(VisionQnABase):
    model_name: str = "llamavision"
    format: str = "llama3"
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, None, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).to(self.device).eval()
    
        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        input_ids = self.model.tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt").unsqueeze(0).to(self.device)
        image_inputs = self.model.processor(
            images=images,
            return_tensors="pt",
            do_resize=True,
            size={"height": 384, "width": 384},
        )

        image_inputs = image_inputs["pixel_values"].to(
            device=self.device, dtype=self.dtype
        )

        image_forward_outs = self.model.vision_model(
            image_inputs,
            output_hidden_states=True,
        )

        image_features = image_forward_outs.hidden_states[-2]

        projected_embeddings = self.model.mm_projector(image_features).to(self.device)

        embedding_layer = self.model.text_model.get_input_embeddings() # .to(self.device)

        new_embeds, attn_mask = self.model.process_tensors(
            input_ids, projected_embeddings, embedding_layer
        )

        default_params = dict(
            temperature=0.2,
            do_sample=True,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        params = self.get_generation_params(request, default_params=default_params)

        output = self.model.text_model.generate(
            inputs_embeds=new_embeds.to(self.device),
            attention_mask=attn_mask.to(self.device),
            **params,
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response
