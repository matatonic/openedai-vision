#!/usr/bin/env python
import io
import requests
import argparse
from datauri import DataURI
from openai import OpenAI

# Initialize argparse
parser = argparse.ArgumentParser(description='Test vision using OpenAI')
parser.add_argument('image_url', type=str, help='URL or image file to be tested')
parser.add_argument('questions', type=str, nargs='*', help='The question to ask the image')
args = parser.parse_args()

client = OpenAI(base_url='http://localhost:5006/v1', api_key='skip')

image_url = args.image_url



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


if not image_url.startswith('http'):
  image_url = str(DataURI.from_file(image_url))

messages = [ { "role": "user", "content": [
    { "type": "text", "text": ' '.join(args.questions) },
    {"type": "image_url", "image_url": { "url": image_url } }
  ]}]

while True:
  response = client.chat.completions.create(model="gpt-4-vision-preview", messages=messages, max_tokens=512,)
  print(f"Answer: {response.choices[0].message.content}\n")
  
  image_url = None
  try:
    q = input("Question: ")
    if q.startswith('http') or q.startswith('data:') or q.startswith('file:'):
      image_url = url_to_data_url(q)
      q = input("Question: ")
  except EOFError as e:
    break
  
  messages.extend([{ "role": "assistant", "content": [ { 'type': 'text', 'text': response.choices[0].message.content } ] },
    { "role": "user", "content": [ { 'type': 'text', 'text': q } ] }
  ])

  if image_url:
    # prepend the new image to the user message
    messages[-1]['content'] = [ {"type": "image_url", "image_url": { "url": image_url } },
      messages[-1]['content'][0] ]


