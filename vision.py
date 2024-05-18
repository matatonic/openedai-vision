#!/usr/bin/env python3
import gc
import os
import sys
import time
import argparse
import importlib
from contextlib import asynccontextmanager
import uvicorn

import openedai
import torch
from vision_qna import *

@asynccontextmanager
async def lifespan(app):
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = openedai.OpenAIStub(lifespan=lifespan)


@app.post(path="/v1/chat/completions")
async def vision_chat_completions(request: ImageChatRequest):

    text = await vision_qna.chat_with_images(request)

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
    t_id = int(time.time())
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

    parser.add_argument('-m', '--model', action='store', default=None, help="The model to use, Ex. llava-hf/llava-v1.6-mistral-7b-hf", required=True)
    parser.add_argument('-b', '--backend', action='store', default=None, help="Force the backend to use (moondream1, moondream2, llavanext, llava, qwen-vl)")
    parser.add_argument('-f', '--format', action='store', default=None, help="Force a specific chat format. (vicuna, mistral, chatml, llama2, phi15, gemma) (doesn't work with all models)")
    parser.add_argument('-d', '--device', action='store', default="auto", help="Set the torch device for the model. Ex. cpu, cuda:1")
    parser.add_argument('--device-map', action='store', default=os.environ.get('OPENEDAI_DEVICE_MAP', "auto"), help="Set the default device map policy for the model. (auto, balanced, sequential, balanced_low_0, cuda:1, etc.)")
    parser.add_argument('--max-memory', action='store', default=None, help="(emu2 only) Set the per cuda device_map max_memory. Ex. 0:22GiB,1:22GiB,cpu:128GiB")
    parser.add_argument('--no-trust-remote-code', action='store_true', help="Don't trust remote code (required for many models)")
    parser.add_argument('-4', '--load-in-4bit', action='store_true', help="load in 4bit (doesn't work with all models)")
    parser.add_argument('-8', '--load-in-8bit', action='store_true', help="load in 8bit (doesn't work with all models)")
    parser.add_argument('-F', '--use-flash-attn', action='store_true', help="Use Flash Attention 2 (doesn't work with all models or GPU)")
    parser.add_argument('-P', '--port', action='store', default=5006, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='0.0.0.0', help="Host to listen on, Ex. localhost")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.model in ['01-ai/Yi-VL-34B', '01-ai/Yi-VL-6B']:
        if False:
            # 💩 fake wrapper for compatibility... but it doesn't work anyways?
            # OSError: Incorrect path_or_model_id: '01-ai/Yi-VL-6B/vit/clip-vit-H-14-laion2B-s32B-b79K-yi-vl-6B-448'. Please provide either the path to a local folder or the repo_id of a model on the Hub.
            
            os.chdir("Yi/VL")
            os.environ['PYTHONPATH'] = '.'
            os.system(f"huggingface-cli download --quiet {args.model} --local-dir {args.model}")
            os.execvp("python", ["python", "openai_api.py", "--model-path", args.model, "--port", f"{args.port}", "--host", args.host])
            sys.exit(0) # not reached
        else:
            os.system(f"huggingface-cli download --quiet {args.model} --local-dir {args.model}")
    
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
    if args.max_memory:
        dev_map_max_memory = {int(dev_id) if dev_id not in ['cpu', 'disk'] else dev_id: mem for dev_id, mem in [dev_mem.split(':') for dev_mem in args.max_memory.split(',')]}
        extra_params['max_memory'] = dev_map_max_memory
    
    vision_qna = backend.VisionQnA(args.model, args.device, args.device_map, extra_params, format=args.format)

    if args.preload or vision_qna is None:
        sys.exit(0)
        
    app.register_model('gpt-4-vision-preview', args.model)

    uvicorn.run(app, host=args.host, port=args.port)
