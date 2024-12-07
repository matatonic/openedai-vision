from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from vision_qna import *

# lmms-lab/llava-onevision-qwen2-0.5b-ov
# lmms-lab/llava-onevision-qwen2-0.5b-si
# lmms-lab/llava-onevision-qwen2-7b-ov
# lmms-lab/llava-onevision-qwen2-7b-si
# lmms-lab/llava-onevision-qwen2-72b-ov
# lmms-lab/llava-onevision-qwen2-72b-si

# BAAI/Aquila-VL-2B-llava-qwen

import warnings
warnings.filterwarnings("ignore")

class VisionQnA(VisionQnABase):
    model_name: str = "llavanextgit" # llava_qwen
    format: str = 'chatml' # qwen_1_5
    vision_layers: List[str] = ["vision_model", "vision_tower", "multi_modal_projector"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):

        load_in_4bit = extra_params.get('load_in_4bit', False)
        load_in_8bit = extra_params.get('load_in_8bit', False)
        if load_in_4bit: del extra_params['load_in_4bit']
        if load_in_8bit: del extra_params['load_in_8bit']

        super().__init__(model_id, device, device_map, extra_params, format)

        if not format:
            self.format = guess_model_format(model_id)

        for i in ['pretrained_model_name_or_path', 'trust_remote_code', 'low_cpu_mem_usage', 'torch_dtype']:
            del self.params[i]

        self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(model_id, None, "llava_qwen", load_in_4bit, load_in_8bit, **self.params)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await prompt_from_messages(request.messages, self.format)

        if len(images) < 1:
            images = [ await url_to_image(black_pixel_url) ]
            prompt = "<image>\n" + prompt

        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size for image in images]

        default_params = dict(
            #pad_token_id=self.processor.tokenizer.eos_token_id,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=4096,
        )

        params = self.get_generation_params(request, default_params)

        generation_kwargs = dict(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break

