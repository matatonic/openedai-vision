#!/usr/bin/env python
import time
import sys
import requests
import argparse
import subprocess
from datauri import DataURI
from openai import OpenAI
import torch

model_tests = [ 
    # fp16
    ["vikhyatk/moondream2", "--use-flash-attn"],
    ["qnguyen3/nanoLLaVA", "--use-flash-attn", "-d", "cuda:0"],
    ["echo840/Monkey"],
    ["echo840/Monkey-Chat"],
    ["internlm/internlm-xcomposer2-7b", "--use-flash-attn", '-d', 'cuda:0'],
    ["internlm/internlm-xcomposer2-vl-7b", "--use-flash-attn", "-d", "cuda:0"],
    ["openbmb/MiniCPM-V", "--use-flash-attn", '-d', 'cuda:0'],
    ["Qwen/Qwen-VL-Chat"],
    ["vikhyatk/moondream1"], # broken
    ["YanweiLi/Mini-Gemini-2B"],
    #["YanweiLi/Mini-Gemini-8x7B"], # requires transformers 4.36.2
    #["deepseek-ai/deepseek-vl-1.3b-chat"], # WIP
    #["deepseek-ai/deepseek-vl-7b-chat"], # WIP
    #["openbmb/OmniLMM-12B"], # WIP
    ["llava-hf/bakLlava-v1-hf", "--use-flash-attn"], # broken
    ["llava-hf/llava-1.5-7b-hf", "--use-flash-attn"],
    ["llava-hf/llava-1.5-13b-hf", "--use-flash-attn"],
    ["llava-hf/llava-v1.6-mistral-7b-hf", "--use-flash-attn"],
    ["llava-hf/llava-v1.6-vicuna-7b-hf", "--use-flash-attn"],
    ["llava-hf/llava-v1.6-vicuna-13b-hf", "--use-flash-attn"],
    ["llava-hf/llava-v1.6-34b-hf", "--use-flash-attn"],
    # 4bit
    ["qnguyen3/nanoLLaVA", "--use-flash-attn", "--load-in-4bit"],
    ["internlm/internlm-xcomposer2-7b-4bit", "--use-flash-attn"], # not recommended, bad quant.
    ["internlm/internlm-xcomposer2-vl-7b-4bit", "--use-flash-attn"],
    ["llava-hf/bakLlava-v1-hf", "--load-in-4bit", "--use-flash-attn"], # broken
    ["llava-hf/llava-1.5-7b-hf", "--load-in-4bit", "--use-flash-attn"],
    ["llava-hf/llava-1.5-13b-hf", "--load-in-4bit", "--use-flash-attn"],
    ["llava-hf/llava-v1.6-mistral-7b-hf", "--load-in-4bit", "--use-flash-attn"],
    ["llava-hf/llava-v1.6-vicuna-7b-hf", "--load-in-4bit", "--use-flash-attn"],
    ["llava-hf/llava-v1.6-vicuna-13b-hf", "--load-in-4bit", "--use-flash-attn"],
    ["llava-hf/llava-v1.6-34b-hf", "--load-in-4bit", "--use-flash-attn"],
]

all_results = []

urls = {
    'tree': 'https://images.freeimages.com/images/large-previews/e59/autumn-tree-1408307.jpg',
    'waterfall': 'https://images.freeimages.com/images/large-previews/242/waterfall-1537490.jpg',
    'horse': 'https://images.freeimages.com/images/large-previews/5fa/attenborough-nature-reserve-1398791.jpg',
    'leaf': 'https://images.freeimages.com/images/large-previews/cd7/gingko-biloba-1058537.jpg',
}

green_pass = '\033[92mpass\033[0m'
red_fail = '\033[91mfail\033[0m'


def data_url_from_url(img_url: str) -> str:
    response = requests.get(img_url)
    
    img_data = response.content
    content_type = response.headers['content-type']
    return str(DataURI.make(mimetype=content_type, charset='utf-8', base64=True, data=img_data))

def ready():
    client = OpenAI(base_url='http://localhost:5006/v1', api_key='skip')
    try:
        return len(client.models.list(timeout=1.0).data) > 0
    except:
        return False

def record_result(cmd_args, results, t, mem, note):
    # update all_results with the test data
    all_results.extend([{
        'args': cmd_args,
        'results': results,
        'time': t,
        'mem': mem,
        'note': note
    }])
    result = all(results)
    print(f"\n#CLI_COMMAND={cmd_args} # test {'pass' if result else 'fail'}, time: {t:.1f}s, mem: {mem:.1f}GB, {note}")

    
def get_total_gpu_mem_used():
    device_count = torch.cuda.device_count()
    total_mem_used = 0.0
    for i in range(device_count):
        allocated_memory, total_memory,  = torch.cuda.mem_get_info(device=i)
        total_mem_used += total_memory - allocated_memory
    return total_mem_used / (1024 ** 3)  # convert bytes to gigabytes
    

def test(cmd_args: list[str]) -> int:
    print(f"### Test start")
    print("Launching server", end='', flush=True)

    proc = subprocess.Popen(['python', 'vision.py', '-m'] + cmd_args,
                            stdout=subprocess.DEVNULL if args.quiet else sys.stdout,
                            stderr=subprocess.DEVNULL if args.quiet else sys.stderr)

    note = ''
    results = []

    while not ready():
        if proc.returncode is not None:
            note = 'Error: Server failed to start.'
            record_result(cmd_args, [False], -1, -1, note)
            print(f"\n{note}\nResult: fail")
            return -1
        
        print(".", end='', flush=True)
        time.sleep(1)
        # XXX TODO: timeout

    print("Server Alive, starting test.\n\n###")
    t = time.time()

    try:
        results = single_round()
    except Exception as e:
        note = f'Test failed with Exception: {e}'
        print(f"{note}")
        results = [False]

    t = time.time() - t

    mem = get_total_gpu_mem_used()

    result = all(results)
    if result:
        note = 'All tests passed.'

    print(f"\n\n###\n\nTest complete.\nResult: {green_pass if result else red_fail}, time: {t:.1f}s")
    

    record_result(cmd_args, results, t, mem, note)

    print("Stopping server", end='', flush=True)

    try:
        proc.communicate(timeout=0)
    except:
        pass

    proc.kill()
    proc.wait()
    
    while proc.returncode is None:
        print(".", end='', flush=True)
        time.sleep(1)

    print(f"\n### Test end")

    return 0 if result else 1

if __name__ == '__main__':
    # Initialize argparse
    parser = argparse.ArgumentParser(description='Test vision using OpenAI')
    parser.add_argument('-s', '--system-prompt', type=str, default=None)
    parser.add_argument('-m', '--max-tokens', type=int, default=None)
    parser.add_argument('-t', '--temperature', type=float, default=None)
    parser.add_argument('-p', '--top_p', type=float, default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose")
    parser.add_argument('--abort-on-fail', action='store_true', help="Abort testing on fail.")
    parser.add_argument('--quiet', action='store_true', help="Less test noise.")
    args = parser.parse_args()

    client = OpenAI(base_url='http://localhost:5006/v1', api_key='skip')

    params = {}
    if args.max_tokens is not None:
        params['max_tokens'] = args.max_tokens
    if args.temperature is not None:
        params['temperature'] = args.temperature
    if args.top_p is not None:
        params['top_p'] = args.top_p

    def generate_response(image_url, prompt):

        messages = [{ "role": "system", "content": [{ 'type': 'text', 'text': args.system_prompt }] }] if args.system_prompt else []
        messages.extend([
            { "role": "user", "content": [
                { "type": "image_url", "image_url": { "url": image_url } },
                { "type": "text", "text": prompt },
            ]}])

        response = client.chat.completions.create(model="gpt-4-vision-preview", messages=messages, **params)
        answer = response.choices[0].message.content
        return answer


    def single_round():
        # XXX TODO: timeout
        results = []
        ### Single round
        # url tests
        for name, url in urls.items():
            answer = generate_response(url, "What is the subject of the image?")
            correct = name in answer.lower()
            results.extend([answer])
            if not correct:
                print(f"{name}[url]: fail, got: {answer}")
                if args.abort_on_fail:
                    break
            else:
                print(f"{name}[url]: pass{', got: ' + answer if args.verbose else ''}")

            data_url = data_url_from_url(url)
            answer = generate_response(data_url, "What is the subject of the image?")
            correct = name in answer.lower()
            results.extend([answer])
            if not correct:
                print(f"{name}[data]: fail, got: {answer}")
                if args.abort_on_fail:
                    break
            else:
                print(f"{name}[data]: pass{', got: ' + answer if args.verbose else ''}")

        return results
    

    print(f"### Begin tests. test count: {len(model_tests)}")

    for i, cmd_args in enumerate(model_tests):
        print(f"### Test {i+1}/{len(model_tests)}: {cmd_args}")
        ret = test(cmd_args)
        if ret != 0 and args.abort_on_fail:
            print(f"### Test {i+1}/{len(model_tests)} Failed.")
            break
    
    print(f"### End tests.")

    print("""# This sample.env file can be to set environment variables for the docker-compose.yml
# Copy this file to vision.env and uncomment the model of your choice.
HF_HOME=/app/hf_home
#CUDA_VISIBLE_DEVICES=1,0""")

    for r in all_results:
        cmdl = ' '.join(r['args'])
        result = all(r['results'])
        print(f"#CLI_COMMAND=\"python vision.py -m {cmdl}\"  # test {'pass' if result else 'fail'}, time: {r['time']:.1f}s, mem: {r['mem']:.1f}GB, {r['note']}")
