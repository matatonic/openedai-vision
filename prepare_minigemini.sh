#!/bin/bash
export HF_HOME=hf_home

if [ -z "$(which huggingface-cli)" ]; then
	echo "First install huggingface-hub: pip install huggingface-hub"
	exit 1
fi

ALL_MODELS="2B 7B 7B-HD 13B 13B-HD 34B 34B-HD 8x7B 8x7B-HD"
MODELS=${*:-}

if [ "$MODELS" = "all" ]; then
	MODELS=$ALL_MODELS
elif [ -z "$MODELS" ]; then
	echo "Specify which sizes of models to download for Mini-Gemini (aka. MGM), or 'all' for all."
	echo "Chose from: $ALL_MODELS"
	echo "Example: $0 2B 8x7B-HD"
	echo "Example: $0 all"
	exit 1
fi

# Required
echo "Downloading required vit/clip models..."
huggingface-cli download OpenAI/clip-vit-large-patch14-336 --local-dir model_zoo/OpenAI/clip-vit-large-patch14-336  || exit
huggingface-cli download laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup --local-dir model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup  || exit

for M in $MODELS; do 
	huggingface-cli download YanweiLi/MGM-$M --local-dir YanweiLi/MGM-$M || exit
done
