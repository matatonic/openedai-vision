OpenedAI Vision
---------------

An OpenAI API compatible vision server, it functions like `gpt-4-vision-preview` and lets you chat about the contents of an image.

- Compatible with the OpenAI Vision API (aka "chat with images")
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

Backend Model support:
- [X] [LlavaNext](https://huggingface.co/llava-hf) - (llava-v1.6-mistral-7b-hf, llava-v1.6-34b-hf - llava-v1.6-34b-hf is not working well yet) *(only supports a single image)
- [X] [Llava](https://huggingface.co/llava-hf) - (llava-v1.5-vicuna-7b-hf, llava-v1.5-vicuna-13b-hf, llava-v1.5-bakLlava-7b-hf) *(only supports a single image)
- [X] [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [X] Moondream2 - [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) *(only supports a single image)
- [ ] Moondream1 - [vikhyatk/moondream1](https://huggingface.co/vikhyatk/moondream1)
- [ ] Deepseek-VL - [deepseek-ai/deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)
- [ ] [openbmb/OmniLMM-12B](https://huggingface.co/openbmb/OmniLMM-12B)
- [ ] [echo840/Monkey](https://huggingface.co/echo840/Monkey)
- [ ] ...


Some vision systems include their own OpenAI compatible API server. Also included are some pre-built images and docker-compose for them:
- [X] [THUDM/CogVLM](https://github.com/THUDM/CogVLM) ([cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf), [cogagent-chat-hf](https://huggingface.co/THUDM/cogagent-chat-hf)), `docker-compose.cogvlm.yml` **Recommended for 16GB-40GB GPU**s
- [X] [01-ai](https://huggingface.co/01-ai)/Yi-VL ([Yi-VL-6B](https://huggingface.co/01-ai/Yi-VL-6B), [Yi-VL-34B](https://huggingface.co/01-ai/Yi-VL-34B)), `docker-compose.yi-vl.yml`

Version: 0.4.0

Recent updates:
- Yi-VL and CogVLM (docker containers only)
- new backend: Qwen-VL
- new backend: llava (1.5)
- new backend: llavanext (1.6+)
- multi-turn questions & answers
- chat_with_images.py test tool and code sample
- selectable chat formats (phi15, vicuna, chatml, llama2/mistral)
- flash attention 2, accelerate (device split), bitsandbytes (4bit, 8bit) support


See: [OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)


API Documentation
-----------------

* [OpenAI Vision guide](https://platform.openai.com/docs/guides/vision)

Installation instructions
-------------------------

```shell
# install the python dependencies
pip install -r requirements.txt
# Install backend specific requirements (or select only backends you plan to use)
pip install -r requirements.moondream.txt -r requirements.qwen-vl.txt
# install the package
pip install .
# run the server
python vision.py
```

Usage
-----

```
usage: vision.py [-h] [-m MODEL] [-b BACKEND] [-f FORMAT] [--load-in-4bit] [--load-in-8bit] [--use-flash-attn] [-d DEVICE] [-P PORT] [-H HOST] [--preload]

OpenedAI Vision API Server

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The model to use, Ex. llava-hf/llava-v1.6-mistral-7b-hf (default: vikhyatk/moondream2)
  -b BACKEND, --backend BACKEND
                        The backend to use (moondream1, moondream2, llavanext, llava, qwen-vl) (default: moondream2)
  -f FORMAT, --format FORMAT
                        Force a specific chat format. (vicuna, mistral, chatml, llama2, phi15) (default: None)
  --load-in-4bit        load in 4bit (doesn't work with all models) (default: False)
  --load-in-8bit        load in 8bit (doesn't work with all models) (default: False)
  --use-flash-attn      Use Flash Attention 2 (doesn't work with all models or GPU) (default: False)
  -d DEVICE, --device DEVICE
                        Set the torch device for the model. Ex. cuda:1 (default: auto)
  -P PORT, --port PORT  Server tcp port (default: 5006)
  -H HOST, --host HOST  Host to listen on, Ex. localhost (default: 0.0.0.0)
  --preload             Preload model and exit. (default: False)
```

Docker support
--------------

You can run the server via docker like so:
```shell
docker compose up
```

Sample API Usage
----------------

`chat_with_image.py` has a sample of how to use the API.

Example:
```
$ python chat_with_image.py https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
Answer: This is a beautiful image of a wooden path leading through a lush green field. The path appears to be well-trodden, suggesting it's a popular route for walking or hiking. The sky is a clear blue with some scattered clouds, indicating a pleasant day with good weather. The field is vibrant and seems to be well-maintained, which could suggest it's part of a park or nature reserve. The overall scene is serene and inviting, perfect for a peaceful walk in nature.

Question: Are there any animals in the picture?
Answer: No, there are no animals visible in the picture. The focus is on the path and the surrounding natural landscape. 

Question: 
```

