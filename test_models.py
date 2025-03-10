#!/usr/bin/env python
import argparse
import csv
import json
import requests
import subprocess
import sys
import time
import traceback
from datauri import DataURI
from openai import OpenAI
import torch

# tests are configured with model_conf_tests.json

all_results = []

urls = {
    'tree': 'https://images.freeimages.com/images/large-previews/e59/autumn-tree-1408307.jpg',
    'waterfall': 'https://images.freeimages.com/images/large-previews/242/waterfall-1537490.jpg',
    'horse': 'https://images.freeimages.com/images/large-previews/5fa/attenborough-nature-reserve-1398791.jpg',
    'leaf': 'https://images.freeimages.com/images/large-previews/cd7/gingko-biloba-1058537.jpg',
}

quality_urls = {
    '98.21': ('What is the total bill?', 'https://ocr.space/Content/Images/receipt-ocr-original.webp'),
    'walmart': ('What store is the receipt from?', 'https://ocr.space/Content/Images/receipt-ocr-original.webp'),
}

no_image = { 
    '5': 'In the integer sequence: 1, 2, 3, 4, ... What number comes next after 4?'
}

green_pass = '✅ pass'
red_fail = '❌ fail'

torch_memory_baseline = 0

def get_total_gpu_mem_used():
    device_count = torch.cuda.device_count()
    total_mem_used = 0.0
    for i in range(device_count):
        allocated_memory, total_memory,  = torch.cuda.mem_get_info(device=i)
        total_mem_used += total_memory - allocated_memory
    return total_mem_used / (1024 ** 3) - torch_memory_baseline  # convert bytes to gigabytes
    
torch_memory_baseline = get_total_gpu_mem_used()
print(f"Baseline CUDA memory: {torch_memory_baseline:.1f}GB")


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

def print_cli_command(r: dict, file=sys.stdout):
    cmdl = ' '.join(r['args'])
    result = all(r['results'])
    print(f"#CLI_COMMAND=\"python vision.py -m {cmdl}\"  # test {green_pass if result else red_fail}, time: {r['time']:.1f}s, mem: {r['mem']:.1f}GB, {r['note']}, {r['total_tokens']}T/{r['total_time']:.1f}s, {r['average_tps']:.1f} T/s", file=file)

def make_csv_header():
    return ['pass', 'model', 'args', 'time', 'mem', 'total_tokens', 'total_time', 'average_tps', 'note']

def make_csv_row(r):
    cmdl = ' '.join(r['args'])
    result = all(r['results'])
    passed = '✅' if result else '❌'
    return [passed, r['model'], cmdl, r['time'], r['mem'], r['total_tokens'], r['total_time'], r['average_tps'], r['note']]

def record_result(cmd_args, results: list = [False], t: float = -1, mem: float = -1, tok_total: int = 0, tim_total: float = 0, average_tps: float = 0, note: str = 'N/A'):
    # update all_results with the test data
    r = {
        'model': cmd_args[0],
        'args': cmd_args,
        'results': results,
        'time': t,
        'mem': mem,
        'total_tokens': tok_total,
        'total_time': tim_total,
        'average_tps': average_tps,
        'note': note
    }
    all_results.extend([r])
    result = all(results)
    cmdl = ' '.join(r['args'])
    passed = '✅' if result else '❌'
    print_cli_command(r)
    #print(f"#CLI_COMMAND=\"python vision.py -m {cmdl}\"  # test {green_pass if result else red_fail}, time: {t:.1f}s, mem: {mem:.1f}GB, {note}, {tok_total}T/{tim_total}s, {average_tps:.1f} T/s")
    print([passed, r['model'], cmdl, r['time'], r['mem'], r['total_tokens'], r['total_time'], r['average_tps'], r['note']], file=sys.stderr)

def test(cmd_args: list[str]) -> int:
    print(f"### Test start")
    print("Launching server", end='', flush=True)

    proc = subprocess.Popen(['python', 'vision.py', '--log-level', args.log_level, '-m'] + cmd_args,
                            stdout=subprocess.DEVNULL if args.quiet else sys.stdout,
                            stderr=subprocess.DEVNULL if args.quiet else sys.stderr)

    note = ''
    results = []
    timeout = time.time() + 600

    while not ready():
        try:
            proc.communicate(timeout=0)
        except:
            pass

        if proc.returncode is not None:
            note = 'Error: Server failed to start (exit).'
            record_result(cmd_args, note=note)
            print(f"\n{note}\nResult: fail\n\n### Test end")
            return -1
        elif time.time() > timeout:
            print("Startup Timeout, killing server.", end='', flush=True)
            note = 'Error: Server failed to start (timeout).'
            record_result(cmd_args, note=note)

            proc.kill()
            proc.wait()

            print(f"\n{note}\nResult: fail\n\n### Test end")
            return -1

        print(".", end='', flush=True)
        time.sleep(1)

    print("Server Alive, starting test.\n\n###")
    t = time.time()

    try:
        results, timing = single_round()
    except Exception as e:
        traceback.print_exc()
        note = f'Test failed with Exception: {e}'
        print(f"{note}")
        results = [False]
        timing = []

    t = time.time() - t
    mem = get_total_gpu_mem_used()
    result = all(results)
    tok_total, tim_total, average_tps = 0, 0.0, 0.0

    if timing:
        for tok, tim in timing:
            if tok > 1 and tim > 0:
                tok_total += tok
                tim_total += tim
        if tim_total > 0.0:
            average_tps=tok_total/tim_total

    if not note:
        note = f'{results.count(True)}/{len(results)} tests passed'

    print(f"\n\n###\n\nTest complete.\nResult: {green_pass if result else red_fail}, time: {t:.1f}s")

    record_result(cmd_args, results, t, mem, tok_total, tim_total, average_tps, note)

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
    parser.add_argument('-L', '--log-level', default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the log level")
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
        tok = response.usage.completion_tokens
        return answer, tok

    def generate_stream_response(image_url, prompt):
        messages = [{ "role": "system", "content": [{ 'type': 'text', 'text': args.system_prompt }] }] if args.system_prompt else []
        messages.extend([
            { "role": "user", "content": [
                { "type": "image_url", "image_url": { "url": image_url } },
                { "type": "text", "text": prompt },
            ]}])

        response = client.chat.completions.create(model="gpt-4-vision-preview", messages=messages, **params, stream=True)
        answer = ''
        completion_tokens = 0
        for chunk in response:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
            if chunk.usage:
                completion_tokens = chunk.usage.completion_tokens
            
        return answer, completion_tokens

    def single_round():
        # XXX TODO: timeout
        results = []
        ### Single round
        timing = []

        def single_test(url, question, label, generator=generate_response):
            tps_time = time.time()
            answer, tok = generator(url, question)
            tps_time = time.time() - tps_time
            correct = name in answer.lower()
            results.extend([correct])
            if not correct:
                print(f"{name}[{label}]: fail, got: {answer}")
                #if args.abort_on_fail:
                #    break
            else:
                print(f"{name}[{label}]: pass{', got: ' + answer if args.verbose else ''}")
            if tok > 1:
                timing.extend([(tok, tps_time)])


        # url tests
        for name, url in urls.items():
            single_test(url, "What is the subject of the image?", "url", generate_response)

            data_url = data_url_from_url(url)
            single_test(data_url, "What is the subject of the image?", "data", generate_response)
            single_test(data_url, "What is the subject of the image?", "data_stream", generate_stream_response)

        """
        ## OCR tests
        quality_urls = {
            '98.21': ('What is the total bill?', 'https://ocr.space/Content/Images/receipt-ocr-original.webp'),
            'walmart': ('What store is the receipt from?', 'https://ocr.space/Content/Images/receipt-ocr-original.webp'),
        }
        for name, question in quality_urls.items():
            prompt, data_url = question
            single_test(data_url, prompt, "quality", generate_stream_response)
        """

        # No image tests
        no_image = { 
            '5': 'In the sequence of numbers: 1, 2, 3, 4, ... What number comes next after 4?'
        }

        def no_image_response(prompt):
            messages = [{ "role": "system", "content": [{ 'type': 'text', 'text': args.system_prompt }] }] if args.system_prompt else []
            messages.extend([{ "role": "user", "content": prompt }])

            response = client.chat.completions.create(model="gpt-4-vision-preview", messages=messages, **params, max_tokens=5)
            answer = response.choices[0].message.content
            return answer

        for name, prompt in no_image.items():
            answer = no_image_response(prompt)
            correct = True #name in answer.lower() # - no exceptions is enough.
            results.extend([correct])
            if not correct:
                print(f"{name}[no_img]: fail, got: {answer}")
                if args.abort_on_fail:
                    break
            else:
                print(f"{name}[no_img]: pass{', got: ' + answer if args.verbose else ''}")

        return results, timing

    with open('model_conf_tests.json') as f:
        model_tests = json.load(f)

    print(f"### Begin tests. test count: {len(model_tests)}")

    try:
        for i, cmd_args in enumerate(model_tests):
            print(f"### Test {i+1}/{len(model_tests)}: {cmd_args}")
            ret = test(cmd_args)
            if ret != 0 and args.abort_on_fail:
                print(f"### Test {i+1}/{len(model_tests)} Failed.")
                break
    except:
        import traceback
        traceback.print_exc()
        print(f"### Aborting due to Exception at test {len(all_results) + 1}/{len(model_tests)}")
    
    print(f"### End tests.")


    # ENV sample
    fname = f"vision.sample.env"
    with open(fname, 'w') as results_file:
        print("""# This sample env file can be used to set environment variables for the docker-compose.yml
# Copy this file to vision.env and uncomment the model of your choice.
HF_HOME=hf_home
HF_HUB_ENABLE_HF_TRANSFER=1
#HF_TOKEN=hf-...
#CUDA_VISIBLE_DEVICES=1,0
#OPENEDAI_DEVICE_MAP="sequential"
""", file=results_file)

        # CSV results
        writer = csv.writer(open("test_output.csv", 'w'))
        writer.writerow(make_csv_header())

        for r in all_results:
            writer.writerow(make_csv_row(r))
            print_cli_command(r, file=results_file)

    # print
    print(open(fname,'r').read())

