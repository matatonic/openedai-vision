from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms
import torch
from vision_qna import *

# THUDM/glm-4v-9b

class VisionQnA(VisionQnABase):
    revision: str = "ade85af5ed77b437edf3cf4d941116026159a618" # until transformers 4.42 support
    model_name: str = "glm-4v"
    format: str = 'glm-4v'
    vision_layers: List[str] = ['vision']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()
        
        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.model.config.vision_config['image_size'], self.model.config.vision_config['image_size']), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, prompt = await glm4v_prompt_from_messages(request.messages)

        input_ids = self.tokenizer.encode(prompt)
        inputs = self.tokenizer.batch_encode_plus(
            [input_ids],
            padding=False,
            truncation=False,
            max_length=None,
            return_tensors="pt",
            is_split_into_words=True,
            add_special_tokens=False
        )

        if images:
            inputs["images"] = torch.stack([ self.transform(img) for img in images ])

        inputs = inputs.to(device=self.device)

        default_params = {
            'max_new_tokens': 2500,
            'do_sample': False,
        }

        params = self.get_generation_params(request, default_params)

#        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=False, skip_prompt=True)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
