
print("deepseek is a WORK IN PROGRESS and doesn't work yet.")

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# specify the path to the model
# model_path = "deepseek-ai/deepseek-vl-7b-chat"

class VisionQnA(VisionQnABase):
    model_name: str = "deepseek-vl"
    format: str = ''
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.processor = VLChatProcessor.from_pretrained(model_id)
        self.model = MultiModalityCausalLM.from_pretrained(**self.params)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")

    async def chat_with_images(self, request: ImageChatRequest) -> str:
        # XXX WIP
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>Describe each stage of this image.",
                "images": ["./images/training_pipelines.jpg"]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)

