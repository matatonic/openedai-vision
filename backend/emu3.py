from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from Emu3.emu3.mllm.processing_emu3 import Emu3Processor

from vision_qna import *

# BAAI/Emu3-Chat

VQ_HUB = "BAAI/Emu3-VisionTokenizer"

class VisionQnA(VisionQnABase):
    model_name: str = "emu3"
    format: str = "vicuna"
    visual_layers: List[str] = []
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
        image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map=self.params['device_map'], trust_remote_code=self.params.get('trust_remote_code', False)).eval()
        self.processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()

    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        image = None
        text = ''

        for m in request.messages:
            if m.role == 'user':
                for c in m.content:
                    if c.type == 'image_url':
                        image = await url_to_image(c.image_url.url)
                        break

        text = "".join([t.text for t in request.messages[-1].content if t.text])

        inputs = self.processor(text=text, image=image, mode='U', padding_side="left", padding="longest", return_tensors="pt")

        default_params = dict(
            max_new_tokens=320,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        params = self.get_generation_params(request, default_params=default_params)

        generation_kwargs = dict(
            input_ids=inputs.input_ids.to(self.device),
            generation_config=GenerationConfig(**params),
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.processor.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.processor.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
