#!/usr/bin/env python3
import sys
import time
import argparse
import importlib

import uvicorn

import openedai
from vision_qna import *


app = openedai.OpenAIStub()

@app.post(path="/v1/chat/completions")
async def vision_chat_completions(request: ImageChatRequest):

    text = await vision_qna.chat_with_images(request.messages, max_tokens=request.max_tokens)

    choices = [ {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "logprobs": None,
            "finish_reason": "stop"
        }
    ]
    t_id = int(time.time() * 1e9)
    vis_chat_resp = {
        "id": f"chatcmpl-{t_id}",
        "object": "chat.completion",
        "created": t_id,
        "model": vision_qna.model_name,
        "system_fingerprint": "fp_111111111",
        "choices": choices,
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

    parser.add_argument('-m', '--model', action='store', default="vikhyatk/moondream2", help="The model to use, Ex. llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument('-b', '--backend', action='store', default=None, help="Force the backend to use (moondream1, moondream2, llavanext, llava, qwen-vl)")
    parser.add_argument('-f', '--format', action='store', default=None, help="Force a specific chat format. (vicuna, mistral, chatml, llama2, phi15, gemma) (doesn't work with all models)")
    parser.add_argument('-d', '--device', action='store', default="auto", help="Set the torch device for the model. Ex. cuda:1")
    parser.add_argument('--no-trust-remote-code', action='store_true', help="Don't trust remote code (required for some models)")
    parser.add_argument('-4', '--load-in-4bit', action='store_true', help="load in 4bit (doesn't work with all models)")
    parser.add_argument('-8', '--load-in-8bit', action='store_true', help="load in 8bit (doesn't work with all models)")
    parser.add_argument('-F', '--use-flash-attn', action='store_true', help="Use Flash Attention 2 (doesn't work with all models or GPU)")
    parser.add_argument('-P', '--port', action='store', default=5006, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='0.0.0.0', help="Host to listen on, Ex. localhost")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if not args.backend:
        args.backend = guess_backend(args.model)

    print(f"Loading VisionQnA[{args.backend}] with {args.model}")
    backend = importlib.import_module(f'backend.{args.backend}')

    extra_params = {}
    if args.load_in_4bit:
        extra_params['load_in_4bit'] = True
    if args.load_in_8bit:
        extra_params['load_in_8bit'] = True
    if args.use_flash_attn:
        extra_params['use_flash_attn'] = True
    
    extra_params['trust_remote_code'] = not args.no_trust_remote_code
    
    vision_qna = backend.VisionQnA(args.model, args.device, extra_params, format=args.format)

    if args.preload:
        sys.exit(0)
        
    app.register_model('gpt-4-vision-preview', args.model)

    uvicorn.run(app, host=args.host, port=args.port)
