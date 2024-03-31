#!/usr/bin/env python
import argparse
from datauri import DataURI
from openai import OpenAI

# Initialize argparse
parser = argparse.ArgumentParser(description='Test vision using OpenAI')
parser.add_argument('image_url', type=str, help='URL or image file to be tested')
parser.add_argument('question', type=str, nargs='?', default='Describe the image', help='The question to ask the image')
args = parser.parse_args()

client = OpenAI(base_url='http://localhost:5006/v1', api_key='skip')

image_url = args.image_url
question = args.question

if not image_url.startswith('http'):
    image_url = str(DataURI.from_file(image_url))

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": question},
        {
          "type": "image_url",
          "image_url": {
            "url": image_url,
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0].message.content)