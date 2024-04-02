OpenedAI Vision
---------------

An OpenAI API compatible vision server, it functions like `gpt-4-vision-preview` and lets you chat about the contents of an image.

- Compatible with the OpenAI Vision API (aka "chat with images")
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

Backend Model support:
- [X] Moondream2 [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) *(only supports a single image)
- [ ] Moondream1 [vikhyatk/moondream1](https://huggingface.co/vikhyatk/moondream1) *(broken for me)
- [X] LlavaNext [llava-v1.6-mistral-7b-hf, llava-v1.6-34b-hf (llava-v1.6-34b-hf is not working well yet)](https://huggingface.co/llava-hf) *(only supports a single image)
- [X] Llava [llava-v1.5-vicuna-7b-hf, llava-v1.5-vicuna-13b-hf, llava-v1.5-bakLlava-7b-hf](https://huggingface.co/llava-hf) *(only supports a single image)
- [ ] Deepseek-VL - [deepseek-ai/deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)
- [ ] ...

Version: 0.3.0

Recent updates:
- llava (1.5) / llavanext (1.6+) backends
- multi-turn questions & answers
- chat_with_images.py test tool
- selectable chat formats (phi15, vicuna, chatml, llama2/mistral)
- flash attention 2, accelerate, bitsandbytes (4bit, 8bit) support


API Documentation
-----------------

* [OpenAI Vision guide](https://platform.openai.com/docs/guides/vision)

Installation instructions
-------------------------

```shell
# install the python dependencies
pip install -r requirements.txt
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
                        The backend to use (moondream1, moondream2, llavanext, llava) (default: moondream2)
  -f FORMAT, --format FORMAT
                        Force a specific chat format. (vicuna, mistral, chatml, llama2, phi15) (default: None)
  --load-in-4bit        load in 4bit (default: False)
  --load-in-8bit        load in 8bit (default: False)
  --use-flash-attn      Use Flash Attention 2 (default: False)
  -d DEVICE, --device DEVICE
                        Set the torch device for the model. Ex. cuda:1 (default: auto)
  -P PORT, --port PORT  Server tcp port (default: 5006)
  -H HOST, --host HOST  Host to listen on, Ex. 0.0.0.0 (default: localhost)
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
