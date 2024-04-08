OpenedAI Vision
---------------

An OpenAI API compatible vision server, it functions like `gpt-4-vision-preview` and lets you chat about the contents of an image.

- Compatible with the OpenAI Vision API (aka "chat with images")
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

Model support:
- [X] [InternLM-XComposer2](https://huggingface.co/internlm/internlm-xcomposer2-7b) [finetune] (multi-image chat model, lots of warnings on startup, wont gpu split)
- [X] [InternLM-XComposer2-VL](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b) [pretrain] *(only supports a single image, also lots of warnings, wont gpu split)
- [X] [LlavaNext](https://huggingface.co/llava-hf) *(only supports a single image)
- - [X] [llava-v1.6-34b-hf](https://huggingface.co/llava-hf/llava-v1.6-34b-hf)
- - [X] [llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)
- - [X] [llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)
- - [X] [llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
- [X] [Llava](https://huggingface.co/llava-hf) *(only supports a single image)
- - [X] [llava-v1.5-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.5-vicuna-7b-hf)
- - [X] [llava-v1.5-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.5-vicuna-13b-hf)
- - [ ] [llava-v1.5-bakLlava-7b-hf](https://huggingface.co/llava-hf/llava-v1.5-bakLlava-7b-hf) (currently errors)
- [X] [Monkey-Chat](https://huggingface.co/echo840/Monkey-Chat)
- [X] [Monkey](https://huggingface.co/echo840/Monkey)
- [X] [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [X] [Moondream2](https://huggingface.co/vikhyatk/moondream2) *(only supports a single image)
- [X] [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V) (aka. OmniLMM-3B) *(only supports a single image)
- [X] [MiniGemini](https://huggingface.co/collections/YanweiLi/) (more complex setup, see: `prepare_minigemini.sh`)
- - [X] [MiniGemini-2B](https://huggingface.co/YanweiLi/Mini-Gemini-2B)
- - [ ] [MiniGemini-7B](https://huggingface.co/YanweiLi/Mini-Gemini-7B) (currently errors)
- - [ ] [MiniGemini-13B](https://huggingface.co/YanweiLi/Mini-Gemini-13B) (currently errors)
- - [ ] [MiniGemini-34B](https://huggingface.co/YanweiLi/Mini-Gemini-34B) (currently errors)
- - [ ] [MiniGemini-8x7B](https://huggingface.co/YanweiLi/Mini-Gemini-8x7B) (currently errors)
- - [ ] [MiniGemini-7B-HD](https://huggingface.co/YanweiLi/Mini-Gemini-7B-HD) (currently errors)
- - [ ] [MiniGemini-13B-HD](https://huggingface.co/YanweiLi/Mini-Gemini-13B-HD) (currently errors)
- - [ ] [MiniGemini-34B-HD](https://huggingface.co/YanweiLi/Mini-Gemini-34B-HD) (currently errors)
- - [ ] [MiniGemini-8x7B-HD](https://huggingface.co/YanweiLi/Mini-Gemini-8x7B-HD) (currently errors)
- [ ] [OmniLMM-12B](https://huggingface.co/openbmb/OmniLMM-12B)
- [ ] [Moondream1](https://huggingface.co/vikhyatk/moondream1)
- [ ] [Deepseek-VL-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)
- [ ] [Deepseek-VL-1.3b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat)
- [ ] [NousResearch/Obsidian-3B-V0.5](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [ ] ...


Some vision systems include their own OpenAI compatible API server. Included are some pre-built images and docker-compose for them (they must be run separately):
- [X] [THUDM/CogVLM](https://github.com/THUDM/CogVLM) `docker-compose.cogvlm.yml`
- - [X] [cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf)
- - [X] [cogagent-chat-hf](https://huggingface.co/THUDM/cogagent-chat-hf) **Recommended for 16GB-40GB GPU**
- [X] [01-ai](https://huggingface.co/01-ai)/Yi-VL `docker-compose.yi-vl.yml`
- - [X] [Yi-VL-6B](https://huggingface.co/01-ai/Yi-VL-6B)
- - [X] [Yi-VL-34B](https://huggingface.co/01-ai/Yi-VL-34B)

Version: 0.7.0

Recent updates:
- new model support: MiniGemini-2B (it's still a bit complex to use, see `prepare_minigemini.sh`)
- new model support: echo840/Monkey-Chat, echo840/Monkey
- AutoGPTQ support for internlm/internlm-xcomposer2-7b-4bit, internlm/internlm-xcomposer2-vl-7b-4bit
- Automatic selection of backend, based on the model name
- Enable trust_remote_code by default
- Improved parameter support: temperature, top_p, max_tokens, system prompts
- Improved default generation parameters and sampler settings
- Improved system prompt for InternLM-XComposer2 & InternLM-XComposer2-VL, Fewer refusals and should not require "In English" nearly as much while still supporting Chinese.
- Fix: chat_with_images.py url filename bug


See: [OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)


API Documentation
-----------------

* [OpenAI Vision guide](https://platform.openai.com/docs/guides/vision)


Docker support
--------------

1) Edit the docker-compose file to suit your needs.

2) You can run the server via docker like so:
```shell
docker compose up
# for CogVLM
docker compose -f docker-compose.cogvlm.yml up
# for VI-VL
docker compose -f docker-compose.yi-vl.yml up
```

Manual Installation instructions
--------------------------------

```shell
# install the python dependencies
pip install -r requirements.txt
# Install backend specific requirements (or select only backends you plan to use)
pip install -r requirements.moondream.txt -r requirements.qwen-vl.txt
# install the package
pip install .
# run the server with your chosen model
python vision.py --model vikhyatk/moondream2
```

For MiniGemini support the docker image is recommended. See `Dockerfile` and `requirements.minigemini.txt` for manual installation instructions.

Usage
-----

```
usage: vision.py [-h] -m MODEL [-b BACKEND] [-f FORMAT] [-d DEVICE] [--no-trust-remote-code] [-4] [-8] [-F] [-P PORT] [-H HOST] [--preload]

OpenedAI Vision API Server

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The model to use, Ex. llava-hf/llava-v1.6-mistral-7b-hf (default: None)
  -b BACKEND, --backend BACKEND
                        Force the backend to use (moondream1, moondream2, llavanext, llava, qwen-vl) (default: None)
  -f FORMAT, --format FORMAT
                        Force a specific chat format. (vicuna, mistral, chatml, llama2, phi15, gemma) (doesn't work with all models) (default: None)
  -d DEVICE, --device DEVICE
                        Set the torch device for the model. Ex. cuda:1 (default: auto)
  c
                        Don't trust remote code (required for some models) (default: False)
  -4, --load-in-4bit    load in 4bit (doesn't work with all models) (default: False)
  -8, --load-in-8bit    load in 8bit (doesn't work with all models) (default: False)
  -F, --use-flash-attn  Use Flash Attention 2 (doesn't work with all models or GPU) (default: False)
  -P PORT, --port PORT  Server tcp port (default: 5006)
  -H HOST, --host HOST  Host to listen on, Ex. localhost (default: 0.0.0.0)
  --preload             Preload model and exit. (default: False)
```

Sample API Usage
----------------

`chat_with_image.py` has a sample of how to use the API.

Example:
```
$ python chat_with_image.py https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
Answer: The image captures a serene landscape of a grassy field, where a wooden walkway cuts through the center. The path is flanked by tall, lush green grass on either side, leading the eye towards the horizon. A few trees and bushes are scattered in the distance, adding depth to the scene. Above, the sky is a clear blue, dotted with white clouds that add to the tranquil atmosphere.


Question: Are there any animals in the picture?
Answer: No, there are no animals visible in the picture.

Question: ^D
$
```

Known Bugs & Workarounds
------------------------

1. Related to cuda device split, If you get:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1! (when checking argument for argument tensors in method wrapper_CUDA_cat)
```
Try to specify a single cuda device with `CUDA_VISIBLE_DEVICES=1` (or # of your GPU) before running the script. or set the device via `--device \<device\>` on the command line.

2. 4bit/8bit and flash attention 2 don't work for all the models. No workaround.
