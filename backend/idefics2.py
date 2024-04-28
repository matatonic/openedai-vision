from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AwqConfig

from vision_qna import *

# "HuggingFaceM4/idefics2-8b"
# "HuggingFaceM4/idefics2-8b-AWQ"

class VisionQnA(VisionQnABase):
    model_name: str = "idefics2"
    
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
            self.params['torch_dtype'] = torch.float16

        self.model = AutoModelForVision2Seq.from_pretrained(**self.params).to(self.device)

        print(f"Loaded on device: {self.model.device} with dtype: {self.model.dtype}")
    
    async def chat_with_images(self, request: ImageChatRequest) -> str:
        images, hfmessages = await images_hfmessages_from_messages(request.messages)

        prompt = self.processor.apply_chat_template(hfmessages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        params = self.get_generation_params(request)
        generated_ids = self.model.generate(**inputs, **params)
        generated_texts = self.processor.decode(generated_ids[0][inputs['input_ids'].size(1):].cpu(), skip_special_tokens=True)

        return generated_texts