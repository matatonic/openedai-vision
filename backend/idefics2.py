from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AwqConfig

from vision_qna import *

# HuggingFaceM4/idefics2-8b
# HuggingFaceM4/idefics2-8b-AWQ
# HuggingFaceM4/idefics2-8b-chatty
# HuggingFaceM4/idefics2-8b-chatty-AWQ

class VisionQnA(VisionQnABase):
    format: str = 'internal'
    model_name: str = "idefics2"
    vision_layers: List[str] = ['vision_model', 'connector']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        #do_image_splitting=False
        #size= {"longest_edge": 448, "shortest_edge": 378} 
        self.processor = AutoProcessor.from_pretrained(model_id)

        if  '-awq' in model_id.lower():
            """
            # This is from https://huggingface.co/HuggingFaceM4/idefics2-8b
            # It doesn't work
            quantization_config = AwqConfig(
                bits=4,
                fuse_max_seq_len=4096,
                modules_to_fuse={
                    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "mlp": ["gate_proj", "up_proj", "down_proj"],
                    "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
                    "use_alibi": False,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "hidden_size": 4096,
                }
            )
            self.params['quantization_config'] = quantization_config
            """

        if self.params['torch_dtype'] == torch.bfloat16:
            self.dtype = self.params['torch_dtype'] = torch.float16

        self.model = AutoModelForVision2Seq.from_pretrained(**self.params)

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
           self.model = self.model.to(self.device)

        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        images, hfmessages = await images_hfmessages_from_messages(request.messages)

        prompt = self.processor.apply_chat_template(hfmessages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images if images else None, return_tensors="pt").to(device=self.model.device)

        # Generate
        params = self.get_generation_params(request)

        generation_kwargs = dict(
            **inputs,
            **params,
        )

        for new_text in threaded_streaming_generator(generate=self.model.generate, tokenizer=self.processor.tokenizer, generation_kwargs=generation_kwargs):
            end = new_text.find(self.processor.tokenizer.eos_token)
            if end == -1:
                yield new_text
            else:
                yield new_text[:end]
                break
