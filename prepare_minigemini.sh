#!/bin/bash
export HF_HOME=hf_home

if [ -z "$(which huggingface-cli)" ]; then
	echo "First install huggingface-hub: pip install huggingface-hub"
	exit 1
fi

echo "Edit this script and uncomment which models to download"

huggingface-cli download OpenAI/clip-vit-large-patch14-336 --local-dir model_zoo/OpenAI/clip-vit-large-patch14-336
huggingface-cli download laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup --local-dir model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup

# Select the model(s) of your choice and download them before starting the server
#huggingface-cli download YanweiLi/Mini-Gemini-2B --local-dir YanweiLi/Mini-Gemini-2B # main image
#huggingface-cli download YanweiLi/Mini-Gemini-7B --local-dir YanweiLi/Mini-Gemini-7B # alt image
#huggingface-cli download YanweiLi/Mini-Gemini-7B-HD --local-dir YanweiLi/Mini-Gemini-7B-HD # alt image
#huggingface-cli download YanweiLi/Mini-Gemini-13B --local-dir YanweiLi/Mini-Gemini-13B # alt image
#huggingface-cli download YanweiLi/Mini-Gemini-13B-HD --local-dir YanweiLi/Mini-Gemini-13B-HD # alt image
#huggingface-cli download YanweiLi/Mini-Gemini-34B --local-dir YanweiLi/Mini-Gemini-34B # alt image
#huggingface-cli download YanweiLi/Mini-Gemini-34B-HD --local-dir YanweiLi/Mini-Gemini-34B-HD # alt image
#huggingface-cli download YanweiLi/Mini-Gemini-8x7B --local-dir YanweiLi/Mini-Gemini-8x7B # alt image
#huggingface-cli download YanweiLi/Mini-Gemini-8x7B-HD --local-dir YanweiLi/Mini-Gemini-8x7B-HD # alt image

#huggingface-cli download 01-ai/Yi-VL-6B --local-dir Yi/VL/01-ai/Yi-VL-6B
#huggingface-cli download 01-ai/Yi-VL-34B --local-dir Yi/VL/01-ai/Yi-VL-34B
