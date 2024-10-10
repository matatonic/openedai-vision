#!/usr/bin/env python
try:
    import dotenv
    load_dotenv(override=True)
except:
    pass

import time
import json
import sys
import os
import requests
import argparse
import subprocess
import traceback
from datauri import DataURI
from openai import OpenAI
import torch

# tests are configured with model_conf_tests.json

all_results = []

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", 'http://localhost:5006/v1'),
    api_key=os.environ.get("OPENAI_API_KEY", 'sk-ip'),
)

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

green_pass = '\033[92mpass\033[0m✅'
red_fail = '\033[91mfail\033[0m❌'


def data_url_from_url(img_url: str) -> str:
    response = requests.get(img_url)
    
    img_data = response.content
    content_type = response.headers['content-type']
    return str(DataURI.make(mimetype=content_type, charset='utf-8', base64=True, data=img_data))

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
    print(f"test {green_pass if result else red_fail}, time: {t:.1f}s, mem: {mem:.1f}GB, {note}")

if __name__ == '__main__':
    # Initialize argparse
    parser = argparse.ArgumentParser(description='Test vision using OpenAI')
    parser.add_argument('-s', '--system-prompt', type=str, default=None)
    parser.add_argument('-m', '--max-tokens', type=int, default=None)
    parser.add_argument('-t', '--temperature', type=float, default=None)
    parser.add_argument('-p', '--top_p', type=float, default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose")
    parser.add_argument('--openai-model', type=str, default="gpt-4-vision-preview")
    parser.add_argument('--abort-on-fail', action='store_true', help="Abort testing on fail.")
    parser.add_argument('--quiet', action='store_true', help="Less test noise.")
    parser.add_argument('-L', '--log-level', default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the log level")
    args = parser.parse_args()


    params = {}
    if args.max_tokens is not None:
        params['max_tokens'] = args.max_tokens
    if args.temperature is not None:
        params['temperature'] = args.temperature
    if args.top_p is not None:
        params['top_p'] = args.top_p

    def generate_response(image_url, prompt):
        messages = [{ "role": "system", "content": [{ 'type': 'text', 'text': args.system_prompt }] }] if args.system_prompt else []

        if isinstance(image_url, str):
            image_url = [image_url]

        content = []
        for url in image_url:
            content.extend([{ "type": "image_url", "image_url": { "url": url } }])
        content.extend([{ "type": "text", "text": prompt }])
        messages.extend([{ "role": "user", "content": content }])

        response = client.chat.completions.create(model=args.openai_model, messages=messages, **params)
        completion_tokens = 0
        answer = response.choices[0].message.content
        if response.usage:
            completion_tokens = response.usage.completion_tokens
        return answer, completion_tokens

    def generate_stream_response(image_url, prompt):
        messages = [{ "role": "system", "content": [{ 'type': 'text', 'text': args.system_prompt }] }] if args.system_prompt else []

        if isinstance(image_url, str):
            image_url = [image_url]

        content = []
        for url in image_url:
            content.extend([{ "type": "image_url", "image_url": { "url": url } }])
        content.extend([{ "type": "text", "text": prompt }])
        messages.extend([{ "role": "user", "content": content }])

        response = client.chat.completions.create(model=args.openai_model, messages=messages, **params, stream=True)
        answer = ''
        completion_tokens = 0
        for chunk in response:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
            if chunk.usage:
                completion_tokens = chunk.usage.completion_tokens

        return answer, completion_tokens

    if True:
        # XXX TODO: timeout
        results = []
        ### Single round
        timing = []

        def single_test(url, question, right_answer, label, generator=generate_response):
            tps_time = time.time()
            answer, tok = generator(url, question)
            tps_time = time.time() - tps_time
            correct = right_answer in answer.lower()
            results.extend([correct])
            if not correct:
                print(f"{right_answer}[{label}]: {red_fail}, got: {answer}")
                #if args.abort_on_fail:
                #    break
            else:
                print(f"{right_answer}[{label}]: {green_pass}{', got: ' + answer if args.verbose else ''}")
            if tok > 1:
                timing.extend([(tok, tps_time)])

        test_time  = time.time()

        # url tests
        for name, url in urls.items():
            single_test(url, "What is the subject of the image?", name, "url", generate_response)

            data_url = data_url_from_url(url)
            single_test(data_url, "What is the subject of the image?", name, "data", generate_response)
            single_test(data_url, "What is the subject of the image?", name, "data_stream", generate_stream_response)


        ## OCR tests
        quality_urls = {
            '98.21': ('What is the total bill?', 'https://ocr.space/Content/Images/receipt-ocr-original.webp'),
            'walmart': ('What store is the receipt from?', 'https://ocr.space/Content/Images/receipt-ocr-original.webp'),
        }
        for name, question in quality_urls.items():
            prompt, data_url = question
            single_test(data_url, prompt, name, "quality", generate_stream_response)

        # No image tests
        no_image = { 
            '5': 'In the sequence of numbers: 1, 2, 3, 4, ... What number comes next after 4? Answer only the number.'
        }

        for name, prompt in no_image.items():
            single_test([], prompt, name, 'no_img', generate_response)

        # Multi-image test
        multi_image = {
            "water": ("What natural element is common in both images?", 
            [ 'https://images.freeimages.com/images/large-previews/e59/autumn-tree-1408307.jpg',
            'https://images.freeimages.com/images/large-previews/242/waterfall-1537490.jpg'])
        }

        for name, question in multi_image.items():
            prompt, data_url = question
            single_test(data_url, prompt, name, "multi-image", generate_stream_response)

        test_time = time.time() - test_time

        result = all(results)
        note = f'{results.count(True)}/{len(results)} tests passed.'
        if timing:
            tok_total, tim_total = 0, 0.0
            for tok, tim in timing:
                if tok > 1 and tim > 0:
                    tok_total += tok
                    tim_total += tim
            if tim_total > 0.0:
                note += f', ({tok_total}/{tim_total:0.1f}s) {tok_total/tim_total:0.1f} T/s'

        print(f"test {green_pass if results else red_fail}, time: {test_time:.1f}s, {note}")
