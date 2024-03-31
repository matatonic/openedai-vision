OpenedAI Vision
---------------

An OpenAI API compatible vision server, it functions like `gpt-4-vision-preview` and lets you chat about the contents of an image.

- Compatible with the OpenAI Vision API (aka "chat with images")
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

Backend Model support:
- [X] Moondream2 [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) *(only a single image and single question currently supported)
- [ ] Deepseek-VL - (in progress) [deepseek-ai/deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)
- [ ] ...

Version: 0.1.0


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
usage: vision.py [-h] [-m MODEL] [-b BACKEND] [-d DEVICE] [-P PORT] [-H HOST] [--preload]

OpenedAI Vision API Server

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The model to use, Ex. deepseek-ai/deepseek-vl-7b-chat (default: vikhyatk/moondream2)
  -b BACKEND, --backend BACKEND
                        The backend to use (moondream, deepseek) (default: moondream)
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

`test_vision.py` has a sample of how to use the API.
Example:
```
$ test_vision.py https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
The image features a long wooden boardwalk running through a lush green field. The boardwalk is situated in a grassy area with trees in the background, creating a serene and picturesque scene. The sky above is filled with clouds, adding to the beauty of the landscape. The boardwalk appears to be a peaceful path for people to walk or hike along, providing a connection between the grassy field and the surrounding environment.
```
