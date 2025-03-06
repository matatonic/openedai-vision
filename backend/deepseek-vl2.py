from transformers import AutoModelForCausalLM

from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images

from vision_qna import *

WIP


class VisionQnA(VisionQnABase):
    model_name: str = "deepseek_vl2"
    format: str = "custom"
    visual_layers: List[str] = []
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        # specify the path to the model
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_id)
        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_id, **self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(device=self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        ## single image conversation example
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n<|ref|>The giraffe at the back.<|/ref|>.",
                "images": ["./images/visual_grounding.jpeg"],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        ## multiple images (or in-context learning) conversation example
        # conversation = [
        #     {
        #         "role": "User",
        #         "content": "<image_placeholder>A dog wearing nothing in the foreground, "
        #                    "<image_placeholder>a dog wearing a santa hat, "
        #                    "<image_placeholder>a dog wearing a wizard outfit, and "
        #                    "<image_placeholder>what's the dog wearing?",
        #         "images": [
        #             "images/dog_a.png",
        #             "images/dog_b.png",
        #             "images/dog_c.png",
        #             "images/dog_d.png",
        #         ],
        #     },
        #     {"role": "Assistant", "content": ""}
        # ]

        conversation = ... prompt

        prepare_inputs = self.processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        default_params = {
            'pad_token_id': self.processor.tokenizer.eos_token_id,
            'bos_token_id': self.processor.tokenizer.bos_token_id,
            'eos_token_id': self.processor.tokenizer.eos_token_id,
            'do_sample': False,
        }

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.processor.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.processor.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
