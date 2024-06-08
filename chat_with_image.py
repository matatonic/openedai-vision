#!/usr/bin/env python
import os
import requests
import argparse
from datauri import DataURI
from openai import OpenAI

try:
    import dotenv
    dotenv.load_dotenv(override=True)
except:
    pass

def url_for_api(img_url: str = None, filename: str = None, always_data=False) -> str:
    if img_url.startswith('http'):
        response = requests.get(img_url)
        
        img_data = response.content
        content_type = response.headers['content-type']
        return str(DataURI.make(mimetype=content_type, charset='utf-8', base64=True, data=img_data))
    elif img_url.startswith('file:'):
        img_url = img_url.replace('file://', '').replace('file:', '')
        return str(DataURI.from_file(img_url))

    return img_url

if __name__ == '__main__':
    # Initialize argparse
    parser = argparse.ArgumentParser(description='Test vision using OpenAI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--system-prompt', type=str, default=None)
    parser.add_argument('-S', '--start-with', type=str, default=None, help="Start reply with, ex. 'Sure, ' (doesn't work with all models)")
    parser.add_argument('-m', '--max-tokens', type=int, default=None)
    parser.add_argument('-t', '--temperature', type=float, default=None)
    parser.add_argument('-p', '--top_p', type=float, default=None)
    parser.add_argument('-u', '--keep-remote-urls', action='store_true', help="Normally, http urls are converted to data: urls for better latency.")
    parser.add_argument('-1', '--single', action='store_true', help='Single turn Q&A, output is only the model response.')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming response.')
    parser.add_argument('image_url', type=str, help='URL or image file to be tested')
    parser.add_argument('questions', type=str, nargs='*', help='The question to ask the image')
    args = parser.parse_args()

    client = OpenAI(base_url=os.environ.get('OPENAI_BASE_URL', 'http://localhost:5006/v1'), api_key='skip')

    params = {}
    if args.max_tokens is not None:
        params['max_tokens'] = args.max_tokens
    if args.temperature is not None:
        params['temperature'] = args.temperature
    if args.top_p is not None:
        params['top_p'] = args.top_p
    params['stream'] = not args.no_stream

    image_url = args.image_url

    if not image_url.startswith('http'):
        image_url = str(DataURI.from_file(image_url))
    elif not args.keep_remote_urls:
        image_url = url_for_api(image_url)

    messages = [{ "role": "system", "content": [{ 'type': 'text', 'text': args.system_prompt }] }] if args.system_prompt else []
    content = [{ "type": "image_url", "image_url": { "url": image_url } },
               { "type": "text", "text": ' '.join(args.questions) }]
    messages.extend([{ "role": "user", "content": content }])

    while True:
        if args.start_with:
            messages.extend([{ "role": "assistant", "content": [{ "type": "text", "text": args.start_with }] }])
        
        response = client.chat.completions.create(model="gpt-4-vision-preview", messages=messages, **params)

        if not args.single:
            print(f"Answer: ", end='', flush=True)
        
        assistant_text = ''

        if args.no_stream:
            assistant_text = response.choices[0].message.content
            print(assistant_text)
        else:
            for chunk in response:
                assistant_text += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end='', flush=True)
            
            print('')

        if args.single:
            break

        image_url = None
        try:
            q = input("\nQuestion: ")

            if q.startswith('http') or q.startswith('data:') or q.startswith('file:'):
                image_url = q
                if image_url.startswith('http') and args.keep_remote_urls:
                    pass
                else:
                    image_url = url_for_api(image_url)

                q = input("Question: ")

        except EOFError as e:
            print('')
            break
        
        content = [{"type": "image_url", "image_url": { "url": image_url } }] if image_url else []
        content.extend([{ 'type': 'text', 'text': assistant_text }])
        messages.extend([{ "role": "assistant", "content": content },
                         { "role": "user", "content": [{ 'type': 'text', 'text': q }] }])


