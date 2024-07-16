from transformers import AutoProcessor, AutoModelForCausalLM

from vision_qna import *

# microsoft/Florence-2-large-ft
# microsoft/Florence-2-base-ft

def select_task(prompt):
    tasks = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>", # simple tasks
        "<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<CAPTION_TO_PHRASE_GROUNDING>",
        "<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>", "<OPEN_VOCABULARY_DETECTION>",
        "<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>", "<OCR_WITH_REGION>"
    ]
    for task in tasks:
        if task in prompt:
            return task
        
    return None

class VisionQnA(VisionQnABase):
    model_name: str = "florence"
    format: str = "florence"
    visual_layers: List[str] = ['vision_tower', 'image_proj_norm', 'image_pos_embed', 'visual_temporal_embed']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()

    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, prompt = await prompt_from_messages(request.messages, self.format)
        
        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]

        inputs = self.processor(text=prompt, images=images[0], return_tensors="pt").to(device=self.model.device, dtype=self.model.dtype)

        default_params = {
            'do_sample': False,
            'num_beams': 3,
        }

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        generated_ids = self.model.generate(**generation_kwargs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=select_task(prompt), image_size=(images[0].width, images[0].height))

        for k, v in parsed_answer.items():
            return str(v)

