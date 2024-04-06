#!/usr/bin/env python
import io
import requests
import argparse
from datauri import DataURI
from openai import OpenAI


def url_to_data_url(img_url: str) -> str:
        if img_url.startswith('http'):
                response = requests.get(img_url)
                
                img_data = response.content
        elif img_url.startswith('data:'):
                return img_url
        elif img_url.startswith('file:'):
                img_url = img_url.replace('file:', '')
                return str(DataURI.from_file(img_url))
        else:
                raise ValueError(f'Unsupported image URL: {img_url}')

        return str(DataURI(io.BytesIO(img_data)))

if __name__ == '__main__':
    # Initialize argparse
    parser = argparse.ArgumentParser(description='Test vision using OpenAI')
    parser.add_argument('-s', '--system-prompt', type=str, default=None)
    parser.add_argument('-m', '--max-tokens', type=int, default=None)
    parser.add_argument('-t', '--temperature', type=float, default=None)
    parser.add_argument('-p', '--top_p', type=float, default=None)
    parser.add_argument('image_url', type=str, help='URL or image file to be tested')
    parser.add_argument('questions', type=str, nargs='*', help='The question to ask the image')
    args = parser.parse_args()

    client = OpenAI(base_url='http://localhost:5006/v1', api_key='skip')

    params = {}
    if args.max_tokens is not None:
        params['max_tokens'] = args.max_tokens
    if args.temperature is not None:
        params['temperature'] = args.temperature
    if args.top_p is not None:
        params['top_p'] = args.top_p

    image_url = args.image_url

    if not image_url.startswith('http'):
        image_url = str(DataURI.from_file(image_url))

    messages = [{ "role": "system", "content": [{ 'type': 'text', 'text': args.system_prompt }] }] if args.system_prompt else []
    content = [{ "type": "image_url", "image_url": { "url": image_url } },
               { "type": "text", "text": ' '.join(args.questions) }]
    messages.extend([{ "role": "user", "content": content }])

    while True:
        response = client.chat.completions.create(model="gpt-4-vision-preview", messages=messages, **params)

        print(f"Answer: {response.choices[0].message.content}\n")
        
        image_url = None
        try:
            q = input("Question: ")
            if q.startswith('http') or q.startswith('data:') or q.startswith('file:'):
                image_url = url_to_data_url(q)
                q = input("Question: ")
        except EOFError as e:
            print('')
            break
        
        content = [{"type": "image_url", "image_url": { "url": image_url } }] if image_url else []
        content.extend([{ 'type': 'text', 'text': response.choices[0].message.content }])
        messages.extend([{ "role": "assistant", "content": content },
                         { "role": "user", "content": [{ 'type': 'text', 'text': q }] }])


