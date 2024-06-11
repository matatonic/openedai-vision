from transformers import AutoModelForCausalLM, AutoTokenizer

import os
from vision_qna import *

# "Qwen/Qwen-VL-Chat" # 13GB
# "Qwen/Qwen-VL-Chat-int4" # 11GB (bad, bugs)

class VisionQnA(VisionQnABase):
    model_name: str = "qwen-vl"
    format: 'chatml'
    vision_layers: List[str] = ['visual']
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModelForCausalLM.from_pretrained(**self.params).eval()

        self.loaded_banner()

    async def chat_with_images(self, request: ImageChatRequest) -> str:
        prompt, history, files, system_prompt = await prompt_history_images_system_from_messages(
            request.messages, img_tok='', url_handler=url_to_file)

        if system_prompt is None:
            system_prompt =  "You are an helpful assistant."
            
        # 1st dialogue turn
        if files:
            query_list = [{'image': files[-1]}, {'text': prompt}]
        else:
            query_list = [{'text': prompt}]

        query = self.tokenizer.from_list_format(query_list)

        default_params = {
            'top_p': 0.3,
        }

        params = self.get_generation_params(request)

        answer, history = self.model.chat(self.tokenizer, query=query, history=history, system=system_prompt, **params)

        for f in files:
            os.remove(f)

        return answer

"""
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        try:
            prompt, history, files, system_prompt = await prompt_history_images_system_from_messages(
                request.messages, img_tok='', url_handler=url_to_file)

            # 1st dialogue turn
            query = self.tokenizer.from_list_format([
                {'image': files[-1] if files else []},
                {'text': prompt},
            ])

            if system_prompt is None:
                system_prompt =  "You are an helpful assistant."

            max_window_size = 16384 # generation_config.max_window_size

# XXX make_context isn't available.
            raw_text, context_tokens = self.model.make_context(
                self.tokenizer,
                query,
                history=history,
                system=system_prompt,
                max_window_size=max_window_size,
                chat_format=self.format,
            )

            input_ids = torch.tensor([context_tokens]).to(self.model.device)

            inputs = dict(
                input_ids=input_ids,
                stop_words_ids=[[self.tokenizer.im_end_id], [self.tokenizer.im_start_id]],
                return_dict_in_generate=False,
            )

            default_params = {
                'top_p': 0.3,
            }

            params = self.get_generation_params(request, default_params=default_params)

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

        except Exception as e:
            logger.error(e)
            # raise

        finally:
            for f in files:
                os.remove(f)
"""
# XXX native streaming doesn't work
"""
  File "/app/backend/qwen-vl.py", line 72, in stream_chat_with_images
    for new_text in streamer:
  File "/app/hf_home/modules/transformers_modules/Qwen/Qwen-VL-Chat/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8/modeling_qwen.py", line 1021, in stream_generator
    for token in self.generate_stream(
                 ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/transformers_stream_generator/main.py", line 208, in generate
    ] = self._prepare_attention_mask_for_generation(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/transformers/generation/utils.py", line 473, in _prepare_attention_mask_for_generation
    torch.isin(elements=inputs, test_elements=pad_token_id).any()
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: isin() received an invalid combination of arguments - got (test_elements=int, elements=Tensor, ), but expected one of:
 * (Tensor elements, Tensor test_elements, *, bool assume_unique, bool invert, Tensor out)
 * (Number element, Tensor test_elements, *, bool assume_unique, bool invert, Tensor out)
 * (Tensor elements, Number test_element, *, bool assume_unique, bool invert, Tensor out)
"""