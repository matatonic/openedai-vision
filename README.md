OpenedAI Vision
---------------

An OpenAI API compatible vision server, it functions like `gpt-4-vision-preview` and lets you chat about the contents of an image.

- Compatible with the OpenAI Vision API (aka "chat with images")
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

## Model support

- [X] [OpenGVLab](https://huggingface.co/OpenGVLab)
- - [X] [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B)
- - [X] [InternVL2-40B](https://huggingface.co/OpenGVLab/InternVL2-40B)
- - [X] [InternVL2-26B](https://huggingface.co/OpenGVLab/InternVL2-26B)
- - [X] [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B)
- - [X] [InternVL2-4B](https://huggingface.co/OpenGVLab/InternVL2-4B) (alternate docker only)
- - [X] [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B)
- - [X] [InternVL2-2B-AWQ](https://huggingface.co/OpenGVLab/InternVL2-2B-AWQ) (currently errors)
- - [X] [InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B)
- - [X] [InternVL-Chat-V1-5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) (wont gpu split yet)
- - [X] [InternVL-Chat-V1-5-Int8](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-Int8) (wont gpu split yet)
- - [X] [Mini-InternVL-Chat-4B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5)
- - [X] [Mini-InternVL-Chat-2B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5)
- [X] [THUDM/CogVLM](https://github.com/THUDM/CogVLM)
- - [X] [cogvlm2-llama3-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B)
- - [X] [cogvlm2-llama3-chinese-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B)
- - [X] [cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf)
- - [X] [cogagent-chat-hf](https://huggingface.co/THUDM/cogagent-chat-hf)
- - [X] [glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b) (wont gpu split)
- [X] [InternLM](https://huggingface.co/internlm/)
- - [X] [XComposer2-2d5-7b](https://huggingface.co/internlm/internlm-xcomposer2d5-7b) (wont gpu split)
- - [X] [XComposer2-4KHD-7b](https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b) (wont gpu split)
- - [X] [XComposer2-7b](https://huggingface.co/internlm/internlm-xcomposer2-7b) [finetune] (wont gpu split)
- - [X] [XComposer2-7b-4bit](https://huggingface.co/internlm/internlm-xcomposer2-7b-4bit) (not recommended)
- - [X] [XComposer2-VL](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b) [pretrain] (wont gpu split)
- - [X] [XComposer2-VL-4bit](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b-4bit)
- - [X] [XComposer2-VL-1.8b](https://huggingface.co/internlm/internlm-xcomposer2-vl-1_8b)
- [X] [HuggingFaceM4/idefics2](https://huggingface.co/HuggingFaceM4) 
- - [X] [idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b) (wont gpu split)
- - [X] [idefics2-8b-AWQ](https://huggingface.co/HuggingFaceM4/idefics2-8b-AWQ) (wont gpu split)
- - [X] [idefics2-8b-chatty](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty) (wont gpu split)
- - [X] [idefics2-8b-chatty-AWQ](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty-AWQ) (wont gpu split)
- [X] [Microsoft](https://huggingface.co/microsoft/)
- - [X] [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
- - [X] [Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft)  (wont gpu split)
- - [X] [Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft)  (wont gpu split)
- [X] [failspy](https://huggingface.co/failspy)
- - [X] [Phi-3-vision-128k-instruct-abliterated-alpha](https://huggingface.co/failspy/Phi-3-vision-128k-instruct-abliterated-alpha)
- [X] [qihoo360](https://huggingface.co/qihoo360)
- - [X] [360VL-8B](https://huggingface.co/qihoo360/360VL-8B)
- - [X] [360VL-70B](https://huggingface.co/qihoo360/360VL-70B) (untested)
- [X] [LlavaNext](https://huggingface.co/llava-hf)
- - [X] [llava-v1.6-34b-hf](https://huggingface.co/llava-hf/llava-v1.6-34b-hf)
- - [X] [llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)
- - [X] [llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)
- - [X] [llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
- [X] [Llava](https://huggingface.co/llava-hf)
- - [X] [llava-v1.5-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.5-vicuna-7b-hf)
- - [X] [llava-v1.5-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.5-vicuna-13b-hf)
- - [ ] [llava-v1.5-bakLlava-7b-hf](https://huggingface.co/llava-hf/llava-v1.5-bakLlava-7b-hf) (currently errors)
- [X] [qresearch](https://huggingface.co/qresearch/)
- - [X] [llama-3-vision-alpha-hf](https://huggingface.co/qresearch/llama-3-vision-alpha-hf) (wont gpu split)
- [X] [BAAI](https://huggingface.co/BAAI/)
- - [X] [BAAI/Bunny-v1_0-2B-zh](https://huggingface.co/BAAI/Bunny-v1_0-2B-zh)
- - [X] [BAAI/Bunny-v1_0-3B-zh](https://huggingface.co/BAAI/Bunny-v1_0-3B-zh)
- - [X] [BAAI/Bunny-v1_0-3B](https://huggingface.co/BAAI/Bunny-v1_0-3B)
- - [X] [BAAI/Bunny-v1_0-4B](https://huggingface.co/BAAI/Bunny-v1_0-4B)
- - [X] [BAAI/Bunny-v1_1-4B](https://huggingface.co/BAAI/Bunny-v1_1-4B)
- - [X] [BAAI/Bunny-v1_1-Llama-3-8B-V](https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V)
- - [X] [Bunny-Llama-3-8B-V](https://huggingface.co/BAAI/Bunny-Llama-3-8B-V)
- - [X] [Emu2-Chat](https://huggingface.co/BAAI/Emu2-Chat) (may need the --max-memory option to GPU split, slow to load)
- [X] [Salesforce](https://huggingface.co/Salesforce)
- - [X] [xgen-mm-phi3-mini-instruct-r-v1](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1) (wont gpu split)
- [X] [TIGER-Lab](https://huggingface.co/TIGER-Lab)
- - [X] [Mantis-8B-siglip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3) (wont gpu split)
- - [X] [Mantis-8B-clip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-clip-llama3) (wont gpu split)
- - [X] [Mantis-8B-Fuyu](https://huggingface.co/TIGER-Lab/Mantis-8B-Fuyu) (wont gpu split)
- [X] [Together.ai](https://huggingface.co/togethercomputer)
- - [X] [Llama-3-8B-Dragonfly-v1](https://huggingface.co/togethercomputer/Llama-3-8B-Dragonfly-v1)
- - [X] [Llama-3-8B-Dragonfly-Med-v1](https://huggingface.co/togethercomputer/Llama-3-8B-Dragonfly-Med-v1) 
- [X] [fuyu-8b](https://huggingface.co/adept/fuyu-8b) [pretrain]
- [X] [falcon-11B-vlm](https://huggingface.co/tiiuae/falcon-11B-vlm)
- [X] [Monkey-Chat](https://huggingface.co/echo840/Monkey-Chat)
- [X] [Monkey](https://huggingface.co/echo840/Monkey)
- [X] [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [X] [Moondream2](https://huggingface.co/vikhyatk/moondream2)
- [X] [Moondream1](https://huggingface.co/vikhyatk/moondream1) (alternate docker only)
- [X] [openbmb](https://huggingface.co/openbmb)
- - [X] [MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)
- - [X] [MiniCPM-V-2](https://huggingface.co/openbmb/MiniCPM-V-2)
- - [X] [MiniCPM-V aka. OmniLMM-3B](https://huggingface.co/openbmb/MiniCPM-V)
- - [ ] [OmniLMM-12B](https://huggingface.co/openbmb/OmniLMM-12B)
- [X] [YanweiLi/MGM](https://huggingface.co/collections/YanweiLi/) (aka Mini-Gemini, more complex setup, see: `prepare_minigemini.sh`)
- - [X] [MGM-2B](https://huggingface.co/YanweiLi/MGM-2B)
- - [X] [MGM-7B](https://huggingface.co/YanweiLi/MGM-7B) (alternate docker only)
- - [X] [MGM-13B](https://huggingface.co/YanweiLi/MGM-13B) (alternate docker only)
- - [X] [MGM-34B](https://huggingface.co/YanweiLi/MGM-34B) (alternate docker only)
- - [X] [MGM-8x7B](https://huggingface.co/YanweiLi/MGM-8x7B) (alternate docker only)
- - [X] [MGM-7B-HD](https://huggingface.co/YanweiLi/MGM-7B-HD) (alternate docker only)
- - [X] [MGM-13B-HD](https://huggingface.co/YanweiLi/MGM-13B-HD) (alternate docker only)
- - [X] [MGM-34B-HD](https://huggingface.co/YanweiLi/MGM-34B-HD) (alternate docker only)
- - [X] [MGM-8x7B-HD](https://huggingface.co/YanweiLi/MGM-8x7B-HD) (alternate docker only)
- [X] [cognitivecomputations]
- - [X] [dolphin-vision-72b](https://huggingface.co/cognitivecomputations/dolphin-vision-72b)
- - [X] [dolphin-vision-7b](https://huggingface.co/cognitivecomputations/dolphin-vision-7b)
- [X] [qnguyen3]
- - [X] [nanoLLaVA](https://huggingface.co/qnguyen3/nanoLLaVA) (wont gpu split)
- - [X] [nanoLLaVA-1.5](https://huggingface.co/qnguyen3/nanoLLaVA-1.5) (wont gpu split)
- [ ] [01-ai/Yi-VL](https://huggingface.co/01-ai)
- - [ ] [Yi-VL-6B](https://huggingface.co/01-ai/Yi-VL-6B) (currently errors)
- - [ ] [Yi-VL-34B](https://huggingface.co/01-ai/Yi-VL-34B) (currently errors)
- [ ] [Deepseek-VL-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)
- [ ] [Deepseek-VL-1.3b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat)
- [ ] [NousResearch/Obsidian-3B-V0.5](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [ ] ...

See: [OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)

## Recent updates

Version 0.28.1

- Update moondream2 support to 2024-07-23
- Pin openbmb/MiniCPM-Llama3-V-2_5 revision

Version 0.28.0

- new model support: internlm-xcomposer2d5-7b
- new model support: dolphin-vision-7b (currently KeyError: 'bunny-qwen')
- Pin glm-v-9B revision until we support transformers 4.42

Version 0.27.1

- new model support: qnguyen3/nanoLLaVA-1.5
- Complete support for chat *without* images (using placeholder images where required, 1x1 clear or 8x8 black as necessary)
- Require transformers==4.41.2 (4.42 breaks many models)

Version 0.27.0

- new model support: OpenGVLab/InternVL2 series of models (1B, 2B, 4B, 8B*, 26B*, 40B*, 76B*) - *(current top open source models)

Version 0.26.0

- new model support: cognitivecomputations/dolphin-vision-72b

Version 0.25.1

- Fix typo in vision.sample.env

Version 0.25.0

- New model support: microsoft/Florence family of models. Not a chat model, but simple questions are ok and all commands are functional. ex "<MORE_DETAILED_CAPTION>", "<OCR>", "<OD>", etc.
- Improved error handling & logging

Version 0.24.1

- Compatibility: Support generation without images for most models. (llava based models still require an image)

Version 0.24.0

- Full streaming support for almost all models.
- Update vikhyatk/moondream2 to 2024-05-20 + streaming
- API compatibility improvements, strip extra leading space if present
- Revert: no more 4bit double quant (slower for insignificant vram savings - protest and it may come back as an option)

Version 0.23.0

- New model support: Together.ai's Llama-3-8B-Dragonfly-v1, Llama-3-8B-Dragonfly-Med-v1 (medical image model)
- Compatibility: [web.chatboxai.app](https://web.chatboxai.app/) can now use openedai-vision as an OpenAI API Compatible backend!
- Initial support for streaming (real streaming for some [dragonfly, internvl-chat-v1-5], fake streaming for the rest). More to come.

Version 0.22.0

- new model support: THUDM/glm-4v-9b

Version 0.21.0

- new model support: Salesforce/xgen-mm-phi3-mini-instruct-r-v1
- Major improvements in quality and compatibility for `--load-in-4/8bit` for many models (InternVL-Chat-V1-5, cogvlm2, MiniCPM-Llama3-V-2_5, Bunny, Monkey, ...). Layer skip with quantized loading.

Version 0.20.0

- enable hf_transfer for faster model downloads (over 300MB/s)
- 6 new Bunny models from [BAAI](https://huggingface.co/BAAI): [Bunny-v1_0-3B-zh](https://huggingface.co/BAAI/Bunny-v1_0-3B-zh), [Bunny-v1_0-3B](https://huggingface.co/BAAI/Bunny-v1_0-3B), [Bunny-v1_0-4B](https://huggingface.co/BAAI/Bunny-v1_0-4B), [Bunny-v1_1-4B](https://huggingface.co/BAAI/Bunny-v1_1-4B), [Bunny-v1_1-Llama-3-8B-V](https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V)

Version 0.19.1

- really Fix <|end|> token for Mini-InternVL-Chat-4B-V1-5, thanks again [@Ph0rk0z](https://github.com/Ph0rk0z)

Version 0.19.0

- new model support: tiiuae/falcon-11B-vlm
- add --max-tiles option for InternVL-Chat-V1-5 and xcomposer2-4khd backends. Tiles use more vram for higher resolution, the default is 6 and 40 respectively, but both are trained up to 40. Some context length warnings may appear near the limits of the model.
- Fix <|end|> token for Mini-InternVL-Chat-4B-V1-5, thanks again [@Ph0rk0z](https://github.com/Ph0rk0z)

Version 0.18.0

- new model support: OpenGVLab/Mini-InternVL-Chat-4B-V1-5, thanks [@Ph0rk0z](https://github.com/Ph0rk0z)
- new model support: failspy/Phi-3-vision-128k-instruct-abliterated-alpha

Version 0.17.0

- new model support: openbmb/MiniCPM-Llama3-V-2_5

Version 0.16.1

- Add "start with" parameter to pre-fill assistant response & backend support (doesn't work with all models) - aka 'Sure,' support.

Version 0.16.0

- new model support: microsoft/Phi-3-vision-128k-instruct

Version 0.15.1

- new model support: OpenGVLab/Mini-InternVL-Chat-2B-V1-5

Version 0.15.0

- new model support: cogvlm2-llama3-chinese-chat-19B, cogvlm2-llama3-chat-19B

Version 0.14.1

- new model support: idefics2-8b-chatty, idefics2-8b-chatty-AWQ (it worked already, no code change)
- new model support: XComposer2-VL-1.8B (it worked already, no code change)

Version: 0.14.0

- docker-compose.yml: Assume the runtime supports the device (ie. nvidia)
- new model support: qihoo360/360VL-8B, qihoo360/360VL-70B (70B is untested, too large for me)
- new model support: BAAI/Emu2-Chat, Can be slow to load, may need --max-memory option control the loading on multiple gpus
- new model support: TIGER-Labs/Mantis: Mantis-8B-siglip-llama3, Mantis-8B-clip-llama3, Mantis-8B-Fuyu

Version: 0.13.0

- new model support: InternLM-XComposer2-4KHD
- new model support: BAAI/Bunny-Llama-3-8B-V
- new model support: qresearch/llama-3-vision-alpha-hf

Version: 0.12.1

- new model support: HuggingFaceM4/idefics2-8b, HuggingFaceM4/idefics2-8b-AWQ
- Fix: remove prompt from output of InternVL-Chat-V1-5

Version: 0.11.0

- new model support: OpenGVLab/InternVL-Chat-V1-5, up to 4k resolution, top opensource model
- MiniGemini renamed MGM upstream


## API Documentation

* [OpenAI Vision guide](https://platform.openai.com/docs/guides/vision)


## Docker support

1) Edit the `vision.env` or `vision-alt.env` file to suit your needs. See: `vision.sample.env` for an example.

```shell
cp vision.sample.env vision.env
# OR for alt the version
cp vision-alt.sample.env vision-alt.env
```

2) You can run the server via docker compose like so:
```shell
# for OpenedAI Vision Server
docker compose up
# for OpenedAI Vision Server (alternate, for Mini-Gemini > 2B, uses transformers==4.36.2)
docker compose -f docker-compose.alt.yml up
```

Add the `-d` flag to daemonize. To install as a service, add `--restart unless-stopped`.

3) To update your setup (or download the image before running the server), you can pull the latest version of the image with the following command:
```shell
# for OpenedAI Vision Server
docker compose pull
# for OpenedAI Vision Server (alternate, for Mini-Gemini > 2B, nanollava, moondream1) which uses transformers==4.36.2
docker compose -f docker-compose.alt.yml pull
```

## Manual Installation instructions

```shell
# install the python dependencies
pip install -U -r requirements.txt "transformers==4.41.2" "autoawq>=0.2.5"
# OR install the python dependencies for the alt version
pip install -U -r requirements.txt "transformers==4.36.2"
# run the server with your chosen model
python vision.py --model vikhyatk/moondream2
```

For MiniGemini support the docker image is recommended. See `prepare_minigemini.sh` for manual installation instructions, models for mini_gemini must be downloaded to local directories, not just run from cache.

## Usage

```
usage: vision.py [-h] -m MODEL [-b BACKEND] [-f FORMAT] [-d DEVICE] [--device-map DEVICE_MAP] [--max-memory MAX_MEMORY] [--no-trust-remote-code] [-4]
                 [-8] [-F] [-T MAX_TILES] [-L {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-P PORT] [-H HOST] [--preload]

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
                        Set the torch device for the model. Ex. cpu, cuda:1 (default: auto)
  --device-map DEVICE_MAP
                        Set the default device map policy for the model. (auto, balanced, sequential, balanced_low_0, cuda:1, etc.) (default: auto)
  --max-memory MAX_MEMORY
                        (emu2 only) Set the per cuda device_map max_memory. Ex. 0:22GiB,1:22GiB,cpu:128GiB (default: None)
  --no-trust-remote-code
                        Don't trust remote code (required for many models) (default: False)
  -4, --load-in-4bit    load in 4bit (doesn't work with all models) (default: False)
  -8, --load-in-8bit    load in 8bit (doesn't work with all models) (default: False)
  -F, --use-flash-attn  Use Flash Attention 2 (doesn't work with all models or GPU) (default: False)
  -T MAX_TILES, --max-tiles MAX_TILES
                        Change the maximum number of tiles. [1-55+] (uses more VRAM for higher resolution, doesn't work with all models) (default: None)
  -L {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the log level (default: INFO)
  -P PORT, --port PORT  Server tcp port (default: 5006)
  -H HOST, --host HOST  Host to listen on, Ex. localhost (default: 0.0.0.0)
  --preload             Preload model and exit. (default: False)

```

## Sample API Usage

`chat_with_image.py` has a sample of how to use the API.

Usage
```
usage: chat_with_image.py [-h] [-s SYSTEM_PROMPT] [-S START_WITH] [-m MAX_TOKENS] [-t TEMPERATURE] [-p TOP_P] [-u] [-1] [--no-stream] image_url [questions ...]

Test vision using OpenAI

positional arguments:
  image_url             URL or image file to be tested
  questions             The question to ask the image (default: None)

options:
  -h, --help            show this help message and exit
  -s SYSTEM_PROMPT, --system-prompt SYSTEM_PROMPT
  -S START_WITH, --start-with START_WITH
                        Start reply with, ex. 'Sure, ' (doesn't work with all models) (default: None)
  -m MAX_TOKENS, --max-tokens MAX_TOKENS
  -t TEMPERATURE, --temperature TEMPERATURE
  -p TOP_P, --top_p TOP_P
  -u, --keep-remote-urls
                        Normally, http urls are converted to data: urls for better latency. (default: False)
  -1, --single          Single turn Q&A, output is only the model response. (default: False)
  --no-stream           Disable streaming response. (default: False)

```

Example:
```
$ python chat_with_image.py -1 https://images.freeimages.com/images/large-previews/cd7/gingko-biloba-1058537.jpg "Describe the image."
The image presents a single, large green leaf with a pointed tip and a serrated edge. The leaf is attached to a thin stem, suggesting it's still connected to its plant. The leaf is set against a stark white background, which contrasts with the leaf's vibrant green color. The leaf's position and the absence of other objects in the image give it a sense of isolation. There are no discernible texts or actions associated with the leaf. The relative position of the leaf to the background remains constant as it is the sole object in the image. The image does not provide any information about the leaf's size or the type of plant it belongs to. The leaf's serrated edge and pointed tip might suggest it's from a deciduous tree, but without additional context, this is purely speculative. The image is a simple yet detailed representation of a single leaf.
```

```
$ python chat_with_image.py https://images.freeimages.com/images/large-previews/e59/autumn-tree-1408307.jpg
Answer: The image captures a serene autumn scene. The main subject is a deciduous tree, standing alone on the shore of a calm lake. The tree is in the midst of changing colors, with leaves in shades of orange, yellow, and green. The branches of the tree are bare, indicating that the leaves are in the process of falling. The tree is positioned on the left side of the image, with its reflection visible in the still water of the lake.

The background of the image features a mountain range, which is partially obscured by a haze. The mountains are covered in a dense forest, with trees displaying a mix of green and autumnal colors. The sky above is clear and blue, suggesting a calm and sunny day.

The overall composition of the image places the tree as the focal point, with the lake, mountains, and sky serving as a picturesque backdrop. The image does not contain any discernible text or human-made objects, reinforcing the natural beauty of the scene. The relative positions of the objects in the image create a sense of depth and perspective, with the tree in the foreground, the lake in the middle ground, and the mountains and sky in the background. The image is a testament to the tranquil beauty of nature during the autumn season.

Question: What kind of tree is it?
Answer: Based on the image, it is not possible to definitively identify the species of the tree. However, the tree's characteristics, such as its broad leaves and the way they change color in the fall, suggest that it could be a type of deciduous tree commonly found in temperate regions. Without more specific details or a closer view, it is not possible to provide a more precise identification of the tree species.

Question: Is it a larch?
Answer: The tree in the image could potentially be a larch, which is a type of deciduous conifer. Larches are known for their needle-like leaves that turn yellow and orange in the fall before falling off. However, without a closer view or more specific details, it is not possible to confirm whether the tree is indeed a larch. The image does not provide enough detail to make a definitive identification of the tree species.

Question: ^D
```

## Known Problems & Workarounds

1. Related to cuda device split, If you get:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1! (when checking argument for argument tensors in method wrapper_CUDA_cat)
```
Try to specify a single cuda device with `CUDA_VISIBLE_DEVICES=1` (or # of your GPU) before running the script. or set the device via `--device-map cuda:0` (or `--device cuda:0` in the alt image!) on the command line.

2. 4bit/8bit quantization and flash attention 2 don't work for all the models. No workaround, see: `sample.env` for known working configurations.

3. Yi-VL is currently not working.

4. The default `--device-map auto` doesn't always work well with these models. If you have issues with multiple GPU's, try using `sequential` and selecting the order of your CUDA devices, like so:

```shell
# Example for reversing the order of 2 devices.
CUDA_VISIBLE_DEVICES=1,0 python vision.py -m llava-hf/llava-v1.6-34b-hf --device-map sequential
```

