#!/usr/bin/env python3
import sys
import time
import argparse
import importlib

from typing import Optional, List, Literal
import uvicorn
from pydantic import BaseModel

import openedai


app = openedai.OpenAIStub()

class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto" # auto -> low (512) or high (Nx512) based on res.

class Content(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

class Message(BaseModel):
    role: str
    content: List[Content]

class ImageChatRequest(BaseModel):
    model: str # = "gpt-4-vision-preview"
    messages: List[Message]
    max_tokens: int = 300

@app.post(path="/v1/chat/completions")
async def chat_with_images(request: ImageChatRequest):

    # XXX only single image & prompt for now
    for c in request.messages[0].content:
        if c.image_url:
            image_url = c.image_url.url
        elif c.text:
            prompt = c.text

    text = await vision_qna.single_question(image_url, prompt)

    t_id = int(time.time() * 1e9)

    vis_chat_resp = {
        "id": f"chatcmpl-{t_id}",
        "object": "chat.completion",
        "created": t_id,
        "model": vision_qna.model_name,
        "system_fingerprint": "fp_111111111",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

    return vis_chat_resp

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='OpenedAI Vision API Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', action='store', default="vikhyatk/moondream2", help="The model to use, Ex. deepseek-ai/deepseek-vl-7b-chat")
    parser.add_argument('-b', '--backend', action='store', default="moondream", help="The backend to use (moondream, deepseek)")
    parser.add_argument('-d', '--device', action='store', default="auto", help="Set the torch device for the model. Ex. cuda:1")
    parser.add_argument('-P', '--port', action='store', default=5006, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='localhost', help="Host to listen on, Ex. 0.0.0.0")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f"Loading VisionQnA[{args.backend}] with {args.model}")
    backend = importlib.import_module(f'backend.{args.backend}')
    vision_qna = backend.VisionQnA(args.model, args.device)

    if args.preload:
        sys.exit(0)
        
    app.register_model('gpt-4-vision-preview', args.model)

    uvicorn.run(app, host=args.host, port=args.port)
