#!/usr/bin/env python3
import gc
import os
import sys
import time
import json
import argparse
import importlib
from contextlib import asynccontextmanager
import uvicorn
from sse_starlette import EventSourceResponse
from loguru import logger

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

    request = vision_qna.repack_message_content(request)

    t_id = int(time.time())
    r_id = f"chatcmpl-{t_id}"

    if request.stream:
        def chat_streaming_chunk(content):
            chunk = {
                "id": r_id,
                "object": "chat.completions.chunk",
                "created": t_id,
                "model": vision_qna.model_name,
                "choices": [{
                    "index": 0,
                    "finish_reason": None,
                    "delta": {'role': 'assistant', 'content': content},
                }],
            }
            return chunk

        async def streamer():
            yield {"data": json.dumps(chat_streaming_chunk(''))}
            logger.debug(f"sse_chunk: ['']")

            # TODO: count tokens
            skip_first_space = True
            dat = ''
            async for resp in vision_qna.stream_chat_with_images(request):
                if skip_first_space:
                    skip_first_space = False
                    if resp[:1] == ' ':
                        resp = resp[1:]

                dat += resp
                if not resp or chr(0xfffd) in dat: # partial unicode char
                    continue

                yield {"data": json.dumps(chat_streaming_chunk(dat))}
                logger.debug(f"sse_chunk: {[dat]}")
                dat = ''

            chunk = chat_streaming_chunk(dat)
            chunk['choices'][0]['finish_reason'] = "stop" # XXX
            chunk['usage'] = {
                "prompt_tokens": 1, # XXX
                "completion_tokens": 1, # XXX
                "total_tokens": 1,  # XXX
            }

            yield {"data": json.dumps(chunk)}
            logger.debug(f"sse_chunk: {[dat]} + ['DONE']")

        return EventSourceResponse(streamer())
    # else:

    text = await vision_qna.chat_with_images(request)

    vis_chat_resp = {
        "id": r_id,
        "object": "chat.completion", # chat.completions.chunk for stream
        "created": t_id,
        "model": vision_qna.model_name,
        "system_fingerprint": "fp_111111111",
        "choices": [ {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "logprobs": None,
            "finish_reason": "stop", # XXX
        } ],
        "usage": {
            "prompt_tokens": 0, # XXX
            "completion_tokens": 0, # XXX
            "total_tokens": 0, # XXX
        }
    }

    logger.debug(f'Response: {vis_chat_resp}')

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
    parser.add_argument('-T', '--max-tiles', action='store', default=None, type=int, help="Change the maximum number of tiles. [1-55+] (uses more VRAM for higher resolution, doesn't work with all models)")
    parser.add_argument('-L', '--log-level', default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the log level")
    parser.add_argument('-P', '--port', action='store', default=5006, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='0.0.0.0', help="Host to listen on, Ex. localhost")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if not args.backend:
        args.backend = guess_backend(args.model)

    logger.info(f"Loading VisionQnA[{args.backend}] with {args.model}")
    backend = importlib.import_module(f'backend.{args.backend}')

    extra_params = {}
    if args.load_in_4bit:
        extra_params['load_in_4bit'] = True
    if args.load_in_8bit:
        extra_params['load_in_8bit'] = True
    if args.use_flash_attn:
        extra_params['use_flash_attn'] = True
    if args.max_tiles:
        extra_params['max_tiles'] = args.max_tiles
    
    logger.remove()
    logger.add(sink=sys.stderr, level=args.log_level)

    extra_params['trust_remote_code'] = not args.no_trust_remote_code
    if args.max_memory:
        dev_map_max_memory = {int(dev_id) if dev_id not in ['cpu', 'disk'] else dev_id: mem for dev_id, mem in [dev_mem.split(':') for dev_mem in args.max_memory.split(',')]}
        extra_params['max_memory'] = dev_map_max_memory
    
    vision_qna = backend.VisionQnA(args.model, args.device, args.device_map, extra_params, format=args.format)

    if args.preload or vision_qna is None:
        sys.exit(0)
        
    app.register_model('gpt-4-vision-preview', args.model)

    uvicorn.run(app, host=args.host, port=args.port)
