OpenedAI Vision
---------------

An OpenAI API compatible vision server, it functions like `gpt-4-vision-preview` and lets you chat about the contents of an image.

- Compatible with the OpenAI Vision API (aka "chat with images")
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

## Model support

Can't decide which to use? See the [OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)

<details>
<summary>Full list of supported models</summary>

- [X] [AIDC-AI]()
- - [X] [Ovis1.5-Gemma2-9B](https://huggingface.co/AIDC-AI/Ovis1.5-Gemma2-9B)
- - [X] [Ovis1.5-Llama3-8B](https://huggingface.co/AIDC-AI/Ovis1.5-Llama3-8B)
- [X] [BAAI](https://huggingface.co/BAAI/)
- - [X] [BAAI/Bunny-v1_0-2B-zh](https://huggingface.co/BAAI/Bunny-v1_0-2B-zh)
- - [X] [BAAI/Bunny-v1_0-3B-zh](https://huggingface.co/BAAI/Bunny-v1_0-3B-zh)
- - [X] [BAAI/Bunny-v1_0-3B](https://huggingface.co/BAAI/Bunny-v1_0-3B)
- - [X] [BAAI/Bunny-v1_0-4B](https://huggingface.co/BAAI/Bunny-v1_0-4B)
- - [X] [BAAI/Bunny-v1_1-4B](https://huggingface.co/BAAI/Bunny-v1_1-4B)
- - [X] [BAAI/Bunny-v1_1-Llama-3-8B-V](https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V)
- - [X] [Bunny-Llama-3-8B-V](https://huggingface.co/BAAI/Bunny-Llama-3-8B-V)
- - [X] [Emu2-Chat](https://huggingface.co/BAAI/Emu2-Chat) (may need the --max-memory option to GPU split, slow to load)
- [X] [cognitivecomputations](https://huggingface.co/cognitivecomputations)
- - [X] [dolphin-vision-72b](https://huggingface.co/cognitivecomputations/dolphin-vision-72b) (alternate docker only)
- - [X] [dolphin-vision-7b](https://huggingface.co/cognitivecomputations/dolphin-vision-7b) (alternate docker only)
- [X] [echo840](https://huggingface.co/echo840)
- - [X] [Monkey-Chat](https://huggingface.co/echo840/Monkey-Chat)
- - [X] [Monkey](https://huggingface.co/echo840/Monkey)
- [X] [failspy](https://huggingface.co/failspy)
- - [X] [Phi-3-vision-128k-instruct-abliterated-alpha](https://huggingface.co/failspy/Phi-3-vision-128k-instruct-abliterated-alpha)
- [X] [falcon-11B-vlm](https://huggingface.co/tiiuae/falcon-11B-vlm) (alternate docker only)
- [X] [fancyfeast/joy-caption-pre-alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha) (caption only)
- [X] [fuyu-8b](https://huggingface.co/adept/fuyu-8b) [pretrain]
- [X] [HuggingFaceM4/idefics2](https://huggingface.co/HuggingFaceM4) 
- - [X] [idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b) (wont gpu split)
- - [X] [idefics2-8b-AWQ](https://huggingface.co/HuggingFaceM4/idefics2-8b-AWQ) (wont gpu split)
- - [X] [idefics2-8b-chatty](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty) (wont gpu split)
- - [X] [idefics2-8b-chatty-AWQ](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty-AWQ) (wont gpu split)
- [X] [InternLM](https://huggingface.co/internlm/)
- - [X] [XComposer2-2d5-7b](https://huggingface.co/internlm/internlm-xcomposer2d5-7b) (wont gpu split)
- - [X] [XComposer2-4KHD-7b](https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b) (wont gpu split)
- - [X] [XComposer2-7b](https://huggingface.co/internlm/internlm-xcomposer2-7b) [finetune] (wont gpu split)
- - [X] [XComposer2-7b-4bit](https://huggingface.co/internlm/internlm-xcomposer2-7b-4bit) (not recommended)
- - [X] [XComposer2-VL](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b) [pretrain] (wont gpu split)
- - [X] [XComposer2-VL-4bit](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b-4bit)
- - [X] [XComposer2-VL-1.8b](https://huggingface.co/internlm/internlm-xcomposer2-vl-1_8b)
- [X] [LMMs-Lab](https://huggingface.co/lmms-lab)
- - [X] [llava-onevision-qwen2-0.5b-ov](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov)
- - [X] [llava-onevision-qwen2-7b-ov](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)
- - [X] [llava-onevision-qwen2-72b-ov](https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov)
- [X] [LlavaNext](https://huggingface.co/llava-hf)
- - [X] [llava-v1.6-34b-hf](https://huggingface.co/llava-hf/llava-v1.6-34b-hf)
- - [X] [llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)
- - [X] [llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)
- - [X] [llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) (alternate docker only)
- [X] [Llava](https://huggingface.co/llava-hf)
- - [X] [llava-v1.5-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.5-vicuna-7b-hf)
- - [X] [llava-v1.5-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.5-vicuna-13b-hf)
- - [ ] [llava-v1.5-bakLlava-7b-hf](https://huggingface.co/llava-hf/llava-v1.5-bakLlava-7b-hf) (currently errors)
- [X] [Microsoft](https://huggingface.co/microsoft/)
- - [X] [Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- - [X] [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
- - [X] [Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft)  (wont gpu split)
- - [X] [Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft)  (wont gpu split)
- [X] [Mistral AI](https://huggingface.co/mistralai)
- - [X] [Pixtral-12B](https://huggingface.co/mistralai/Pixtral-12B-2409)
- [X] [omlab/omchat-v2.0-13B-single-beta_hf](https://huggingface.co/omlab/omchat-v2.0-13B-single-beta_hf)
- [X] [openbmb](https://huggingface.co/openbmb)
- - [X] [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) (video not supported yet)
- - [X] [MiniCPM-V-2_6-int4](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4)
- - [X] [MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)
- - [X] [MiniCPM-V-2](https://huggingface.co/openbmb/MiniCPM-V-2) (alternate docker only)
- - [X] [MiniCPM-V aka. OmniLMM-3B](https://huggingface.co/openbmb/MiniCPM-V) (alternate docker only)
- - [ ] [OmniLMM-12B](https://huggingface.co/openbmb/OmniLMM-12B)
- [X] [OpenGVLab](https://huggingface.co/OpenGVLab)
- - [X] [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B)
- - [X] [InternVL2-40B](https://huggingface.co/OpenGVLab/InternVL2-40B)
- - [X] [InternVL2-26B](https://huggingface.co/OpenGVLab/InternVL2-26B)
- - [X] [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B)
- - [X] [InternVL2-4B](https://huggingface.co/OpenGVLab/InternVL2-4B) (alternate docker only)
- - [X] [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B)
- - [ ] [InternVL2-2B-AWQ](https://huggingface.co/OpenGVLab/InternVL2-2B-AWQ) (currently errors)
- - [X] [InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B)
- - [X] [InternVL-Chat-V1-5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) (wont gpu split yet)
- - [ ] [InternVL-Chat-V1-5-AWQ](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-AWQ) (wont gpu split yet)
- - [X] [Mini-InternVL-Chat-4B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5) (alternate docker only)
- - [X] [Mini-InternVL-Chat-2B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5)
- [X] [Salesforce](https://huggingface.co/Salesforce)
- - [X] [xgen-mm-phi3-mini-instruct-singleimage-r-v1.5](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-singleimage-r-v1.5)
- - [X] [xgen-mm-phi3-mini-instruct-interleave-r-v1](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5)
- - [X] [xgen-mm-phi3-mini-instruct-dpo-r-v1.5](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-dpo-r-v1.5)
- - [X] [xgen-mm-phi3-mini-instruct-r-v1](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1) (wont gpu split)
- [X] [THUDM/CogVLM](https://github.com/THUDM/CogVLM)
- - [X] [cogvlm2-llama3-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) (alternate docker only)
- - [X] [cogvlm2-llama3-chinese-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B) (alternate docker only)
- - [X] [cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf) (alternate docker only)
- - [X] [cogagent-chat-hf](https://huggingface.co/THUDM/cogagent-chat-hf) (alternate docker only)
- - [X] [glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b) (wont gpu split)
- [X] [TIGER-Lab](https://huggingface.co/TIGER-Lab)
- - [X] [Mantis-8B-siglip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3) (wont gpu split)
- - [X] [Mantis-8B-clip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-clip-llama3) (wont gpu split)
- - [X] [Mantis-8B-Fuyu](https://huggingface.co/TIGER-Lab/Mantis-8B-Fuyu) (wont gpu split)
- [X] [Together.ai](https://huggingface.co/togethercomputer)
- - [X] [Llama-3-8B-Dragonfly-v1](https://huggingface.co/togethercomputer/Llama-3-8B-Dragonfly-v1)
- - [X] [Llama-3-8B-Dragonfly-Med-v1](https://huggingface.co/togethercomputer/Llama-3-8B-Dragonfly-Med-v1) 
- [X] [qihoo360](https://huggingface.co/qihoo360)
- - [X] [360VL-8B](https://huggingface.co/qihoo360/360VL-8B)
- - [X] [360VL-70B](https://huggingface.co/qihoo360/360VL-70B) (untested)
- [X] [qnguyen3](https://huggingface.co/qnguyen3)
- - [X] [nanoLLaVA](https://huggingface.co/qnguyen3/nanoLLaVA) (wont gpu split)
- - [X] [nanoLLaVA-1.5](https://huggingface.co/qnguyen3/nanoLLaVA-1.5) (wont gpu split)
- [X] [qresearch](https://huggingface.co/qresearch/)
- - [X] [llama-3-vision-alpha-hf](https://huggingface.co/qresearch/llama-3-vision-alpha-hf) (wont gpu split)
- [X] [Qwen](https://huggingface.co/Qwen/)
- - [X] [Qwen2-VL-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-AWQ)
- - [X] [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- - [X] [Qwen2-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-AWQ)
- - [X] [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- - [X] [Qwen2-VL-2B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-AWQ)
- - [X] [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [X] [vikhyatk](https://huggingface.co/vikhyatk)
- - [X] [moondream2](https://huggingface.co/vikhyatk/moondream2)
- - [X] [moondream1](https://huggingface.co/vikhyatk/moondream1) (0.28.1-alt only)
- [X] [YanweiLi/MGM](https://huggingface.co/collections/YanweiLi/) (0.28.1-alt only)
- - [X] [MGM-2B](https://huggingface.co/YanweiLi/MGM-2B) (0.28.1-alt only)
- - [X] [MGM-7B](https://huggingface.co/YanweiLi/MGM-7B) (0.28.1-alt only)
- - [X] [MGM-13B](https://huggingface.co/YanweiLi/MGM-13B) (0.28.1-alt only)
- - [X] [MGM-34B](https://huggingface.co/YanweiLi/MGM-34B) (0.28.1-alt only)
- - [X] [MGM-8x7B](https://huggingface.co/YanweiLi/MGM-8x7B) (0.28.1-alt only)
- - [X] [MGM-7B-HD](https://huggingface.co/YanweiLi/MGM-7B-HD) (0.28.1-alt only)
- - [X] [MGM-13B-HD](https://huggingface.co/YanweiLi/MGM-13B-HD) (0.28.1-alt only)
- - [X] [MGM-34B-HD](https://huggingface.co/YanweiLi/MGM-34B-HD) (0.28.1-alt only)
- - [X] [MGM-8x7B-HD](https://huggingface.co/YanweiLi/MGM-8x7B-HD) (0.28.1-alt only)

</details>

If you can't find your favorite model, you can [open a new issue](https://github.com/matatonic/openedai-vision/issues/new/choose) and request it.

## Recent updates

Version 0.32.0

- new model support: From AIDC-AI, Ovis1.5-Gemma2-9B and Ovis1.5-Llama3-8B
- new model support: omlab/omchat-v2.0-13B-single-beta_hf

Version 0.31.1

- Fix support for openbmb/MiniCPM-V-2_6-int4

Version 0.31.0

- new model support: Qwen/Qwen2-VL family of models (video untested, GPTQ not working yet, but AWQ and BF16 are fine)
- transformers from git
- Regression: THUD/glm-4v-9b broken in this release (re: transformers)

Version 0.30.0

- new model support: mistralai/Pixtral-12B-2409 (no streaming yet, no quants yet)
- new model support: LMMs-Lab's llava-onevision-qwen2, 0.5b, 7b and 72b (72b untested, 4bit support doesn't seem to work properly yet)
- Update moondream2 to version 2024-08-26
- Performance fixed: idefics2-8b-AWQ, idefics2-8b-chatty-AWQ

Version 0.29.0

- new model support: fancyfeast/joy-caption-pre-alpha (caption only, depends on Meta-Llama-3.1-8b [authorization required], --load-in-4bit avoids this dependency)
- new model support: MiniCPM-V-2_6 (video not supported yet)
- new model support: microsoft/Phi-3.5-vision-instruct (worked without any changes)
- new model support: Salesforce/xgen-mm-phi3-mini-instruct-r-v1.5 family of models: singleimage, dpo, interleave
- library updates: torch 2.4, transformers >=4.44.2
- New `-alt` docker image support (transformers==4.41.2, was 4.36.2)
- !!! ⚠️ WARNING ⚠️ !!! Broken in this release: MiniCPM-V, MiniCPM-V-2, llava-v1.6-mistral-7b-hf, internlm-xcomposer2* (all 4bit), dolphin-vision* (all), THUDM/cog* (all), InternVL2-4B,Mini-InternVL-Chat-4B-V1-5, falcon-11B-vlm !!! ⚠️ WARNING ⚠️ !!!
- - Use version `:0.28.1` or the `-alt` docker image for continued support of these models.
- Performance regression: idefics2-8b-AWQ, idefics2-8b-chatty-AWQ
- ⚠️ DEPRECATED MODELS: YanweiLi/MGM*, Moondream1 (use the `-alt:0.28.1` image for support of these models)
- unpin MiniCPM-Llama3-V-2_5, glm-v-9B revisions


<details>
<summary>Older version notes</summary>

Version 0.28.1

- Update moondream2 support to 2024-07-23
- Pin openbmb/MiniCPM-Llama3-V-2_5 revision

Version 0.28.0

- new model support: internlm-xcomposer2d5-7b
- new model support: dolphin-vision-7b (currently KeyError: 'bunny-qwen')
- Pin glm-v-9B revision until we support transformers 4.42

Version 0.27.1

- new model support: qnguyen3/nanoLLaVA-1.5
- Complete support for chat *without* images (using placeholder images where required, 1x1 clear or 8x8 black as necessary)
- Require transformers==4.41.2 (4.42 breaks many models)

Version 0.27.0

- new model support: OpenGVLab/InternVL2 series of models (1B, 2B, 4B, 8B*, 26B*, 40B*, 76B*) - *(current top open source models)

Version 0.26.0

- new model support: cognitivecomputations/dolphin-vision-72b

Version 0.25.1

- Fix typo in vision.sample.env

Version 0.25.0

- New model support: microsoft/Florence family of models. Not a chat model, but simple questions are ok and all commands are functional. ex "<MORE_DETAILED_CAPTION>", "<OCR>", "<OD>", etc.
- Improved error handling & logging

Version 0.24.1

- Compatibility: Support generation without images for most models. (llava based models still require an image)

Version 0.24.0

- Full streaming support for almost all models.
- Update vikhyatk/moondream2 to 2024-05-20 + streaming
- API compatibility improvements, strip extra leading space if present
- Revert: no more 4bit double quant (slower for insignificant vram savings - protest and it may come back as an option)

Version 0.23.0

- New model support: Together.ai's Llama-3-8B-Dragonfly-v1, Llama-3-8B-Dragonfly-Med-v1 (medical image model)
- Compatibility: [web.chatboxai.app](https://web.chatboxai.app/) can now use openedai-vision as an OpenAI API Compatible backend!
- Initial support for streaming (real streaming for some [dragonfly, internvl-chat-v1-5], fake streaming for the rest). More to come.

Version 0.22.0

- new model support: THUDM/glm-4v-9b

Version 0.21.0

- new model support: Salesforce/xgen-mm-phi3-mini-instruct-r-v1
- Major improvements in quality and compatibility for `--load-in-4/8bit` for many models (InternVL-Chat-V1-5, cogvlm2, MiniCPM-Llama3-V-2_5, Bunny, Monkey, ...). Layer skip with quantized loading.

Version 0.20.0

- enable hf_transfer for faster model downloads (over 300MB/s)
- 6 new Bunny models from [BAAI](https://huggingface.co/BAAI): [Bunny-v1_0-3B-zh](https://huggingface.co/BAAI/Bunny-v1_0-3B-zh), [Bunny-v1_0-3B](https://huggingface.co/BAAI/Bunny-v1_0-3B), [Bunny-v1_0-4B](https://huggingface.co/BAAI/Bunny-v1_0-4B), [Bunny-v1_1-4B](https://huggingface.co/BAAI/Bunny-v1_1-4B), [Bunny-v1_1-Llama-3-8B-V](https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V)

Version 0.19.1

- really Fix <|end|> token for Mini-InternVL-Chat-4B-V1-5, thanks again [@Ph0rk0z](https://github.com/Ph0rk0z)

Version 0.19.0

- new model support: tiiuae/falcon-11B-vlm
- add --max-tiles option for InternVL-Chat-V1-5 and xcomposer2-4khd backends. Tiles use more vram for higher resolution, the default is 6 and 40 respectively, but both are trained up to 40. Some context length warnings may appear near the limits of the model.
- Fix <|end|> token for Mini-InternVL-Chat-4B-V1-5, thanks again [@Ph0rk0z](https://github.com/Ph0rk0z)

Version 0.18.0

- new model support: OpenGVLab/Mini-InternVL-Chat-4B-V1-5, thanks [@Ph0rk0z](https://github.com/Ph0rk0z)
- new model support: failspy/Phi-3-vision-128k-instruct-abliterated-alpha

Version 0.17.0

- new model support: openbmb/MiniCPM-Llama3-V-2_5

Version 0.16.1

- Add "start with" parameter to pre-fill assistant response & backend support (doesn't work with all models) - aka 'Sure,' support.

Version 0.16.0

- new model support: microsoft/Phi-3-vision-128k-instruct

Version 0.15.1

- new model support: OpenGVLab/Mini-InternVL-Chat-2B-V1-5

Version 0.15.0

- new model support: cogvlm2-llama3-chinese-chat-19B, cogvlm2-llama3-chat-19B

Version 0.14.1

- new model support: idefics2-8b-chatty, idefics2-8b-chatty-AWQ (it worked already, no code change)
- new model support: XComposer2-VL-1.8B (it worked already, no code change)

Version: 0.14.0

- docker-compose.yml: Assume the runtime supports the device (ie. nvidia)
- new model support: qihoo360/360VL-8B, qihoo360/360VL-70B (70B is untested, too large for me)
- new model support: BAAI/Emu2-Chat, Can be slow to load, may need --max-memory option control the loading on multiple gpus
- new model support: TIGER-Labs/Mantis: Mantis-8B-siglip-llama3, Mantis-8B-clip-llama3, Mantis-8B-Fuyu

Version: 0.13.0

- new model support: InternLM-XComposer2-4KHD
- new model support: BAAI/Bunny-Llama-3-8B-V
- new model support: qresearch/llama-3-vision-alpha-hf

Version: 0.12.1

- new model support: HuggingFaceM4/idefics2-8b, HuggingFaceM4/idefics2-8b-AWQ
- Fix: remove prompt from output of InternVL-Chat-V1-5

Version: 0.11.0

- new model support: OpenGVLab/InternVL-Chat-V1-5, up to 4k resolution, top opensource model
- MiniGemini renamed MGM upstream
</details>

## API Documentation

* [OpenAI Vision guide](https://platform.openai.com/docs/guides/vision)


## Docker support (tested, recommended)

1) Edit the `vision.env` or `vision-alt.env` file to suit your needs. See: `vision.sample.env` for an example.

```shell
cp vision.sample.env vision.env
# OR for alt the version
cp vision-alt.sample.env vision-alt.env
```

2) You can run the server via docker compose like so:
```shell
# for OpenedAI Vision Server
docker compose up
# for OpenedAI Vision Server (alternate)
docker compose -f docker-compose.alt.yml up
```

Add the `-d` flag to daemonize and run in the background as a service.

3) To update your setup (or download the image before running the server), you can pull the latest version of the image with the following command:
```shell
# for OpenedAI Vision Server
docker compose pull
# for OpenedAI Vision Server (alternate)
docker compose -f docker-compose.alt.yml pull
```

## Manual Installation instructions

```shell
# Create & activate a new virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate
# install the python dependencies
pip install -U -r requirements.txt "git+https://github.com/huggingface/transformers" "autoawq>=0.2.5"
# OR install the python dependencies for the alt version
pip install -U -r requirements.txt "transformers==4.41.2"
# run the server with your chosen model
python vision.py --model vikhyatk/moondream2
```

Additional steps may be required for some models, see the Dockerfile for the latest installation instructions.

## Usage

```
usage: vision.py [-h] -m MODEL [-b BACKEND] [-f FORMAT] [-d DEVICE] [--device-map DEVICE_MAP] [--max-memory MAX_MEMORY] [--no-trust-remote-code] [-4] [-8] [-F] [-A {sdpa,eager,flash_attention_2}] [-T MAX_TILES] [--preload]
                 [-L {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-H HOST] [-P PORT]

OpenedAI Vision API Server

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The model to use, Ex. llava-hf/llava-v1.6-mistral-7b-hf (default: None)
  -b BACKEND, --backend BACKEND
                        Force the backend to use (phi3, idefics2, llavanext, llava, etc.) (default: None)
  -f FORMAT, --format FORMAT
                        Force a specific chat format. (vicuna, mistral, chatml, llama2, phi15, etc.) (doesn't work with all models) (default: None)
  -d DEVICE, --device DEVICE
                        Set the torch device for the model. Ex. cpu, cuda:1 (default: auto)
  --device-map DEVICE_MAP
                        Set the default device map policy for the model. (auto, balanced, sequential, balanced_low_0, cuda:1, etc.) (default: auto)
  --max-memory MAX_MEMORY
                        (emu2 only) Set the per cuda device_map max_memory. Ex. 0:22GiB,1:22GiB,cpu:128GiB (default: None)
  --no-trust-remote-code
                        Don't trust remote code (required for many models) (default: False)
  -4, --load-in-4bit    load in 4bit (doesn't work with all models) (default: False)
  -8, --load-in-8bit    load in 8bit (doesn't work with all models) (default: False)
  -F, --use-flash-attn  DEPRECATED: use --attn_implementation flash_attention_2 or -A flash_attention_2 (default: False)
  -A {sdpa,eager,flash_attention_2}, --attn_implementation {sdpa,eager,flash_attention_2}
                        Set the attn_implementation (default: sdpa)
  -T MAX_TILES, --max-tiles MAX_TILES
                        Change the maximum number of tiles. [1-55+] (uses more VRAM for higher resolution, doesn't work with all models) (default: None)
  --preload             Preload model and exit. (default: False)
  -L {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the log level (default: INFO)
  -H HOST, --host HOST  Host to listen on, Ex. localhost (default: 0.0.0.0)
  -P PORT, --port PORT  Server tcp port (default: 5006)

```

## Sample API Usage

`chat_with_image.py` has a sample of how to use the API.

Usage
```
usage: chat_with_image.py [-h] [-s SYSTEM_PROMPT] [--openai-model OPENAI_MODEL] [-S START_WITH] [-m MAX_TOKENS] [-t TEMPERATURE] [-p TOP_P] [-u] [-1] [--no-stream] image_url [questions ...]

Test vision using OpenAI

positional arguments:
  image_url             URL or image file to be tested
  questions             The question to ask the image (default: None)

options:
  -h, --help            show this help message and exit
  -s SYSTEM_PROMPT, --system-prompt SYSTEM_PROMPT
                        Set a system prompt. (default: None)
  --openai-model OPENAI_MODEL
                        OpenAI model to use. (default: gpt-4-vision-preview)
  -S START_WITH, --start-with START_WITH
                        Start reply with, ex. 'Sure, ' (doesn't work with all models) (default: None)
  -m MAX_TOKENS, --max-tokens MAX_TOKENS
                        Max tokens to generate. (default: None)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature. (default: None)
  -p TOP_P, --top_p TOP_P
                        top_p (default: None)
  -u, --keep-remote-urls
                        Normally, http urls are converted to data: urls for better latency. (default: False)
  -1, --single          Single turn Q&A, output is only the model response. (default: False)
  --no-stream           Disable streaming response. (default: False)

```

Example:
```
$ python chat_with_image.py -1 https://images.freeimages.com/images/large-previews/cd7/gingko-biloba-1058537.jpg "Describe the image."
The image presents a single, large green leaf with a pointed tip and a serrated edge. The leaf is attached to a thin stem, suggesting it's still connected to its plant. The leaf is set against a stark white background, which contrasts with the leaf's vibrant green color. The leaf's position and the absence of other objects in the image give it a sense of isolation. There are no discernible texts or actions associated with the leaf. The relative position of the leaf to the background remains constant as it is the sole object in the image. The image does not provide any information about the leaf's size or the type of plant it belongs to. The leaf's serrated edge and pointed tip might suggest it's from a deciduous tree, but without additional context, this is purely speculative. The image is a simple yet detailed representation of a single leaf.
```

```
$ python chat_with_image.py https://images.freeimages.com/images/large-previews/e59/autumn-tree-1408307.jpg
Answer: The image captures a serene autumn scene. The main subject is a deciduous tree, standing alone on the shore of a calm lake. The tree is in the midst of changing colors, with leaves in shades of orange, yellow, and green. The branches of the tree are bare, indicating that the leaves are in the process of falling. The tree is positioned on the left side of the image, with its reflection visible in the still water of the lake.

The background of the image features a mountain range, which is partially obscured by a haze. The mountains are covered in a dense forest, with trees displaying a mix of green and autumnal colors. The sky above is clear and blue, suggesting a calm and sunny day.

The overall composition of the image places the tree as the focal point, with the lake, mountains, and sky serving as a picturesque backdrop. The image does not contain any discernible text or human-made objects, reinforcing the natural beauty of the scene. The relative positions of the objects in the image create a sense of depth and perspective, with the tree in the foreground, the lake in the middle ground, and the mountains and sky in the background. The image is a testament to the tranquil beauty of nature during the autumn season.

Question: What kind of tree is it?
Answer: Based on the image, it is not possible to definitively identify the species of the tree. However, the tree's characteristics, such as its broad leaves and the way they change color in the fall, suggest that it could be a type of deciduous tree commonly found in temperate regions. Without more specific details or a closer view, it is not possible to provide a more precise identification of the tree species.

Question: Is it a larch?
Answer: The tree in the image could potentially be a larch, which is a type of deciduous conifer. Larches are known for their needle-like leaves that turn yellow and orange in the fall before falling off. However, without a closer view or more specific details, it is not possible to confirm whether the tree is indeed a larch. The image does not provide enough detail to make a definitive identification of the tree species.

Question: ^D
```

## Known Problems & Workarounds

1. Related to cuda device split, If you get:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1! (when checking argument for argument tensors in method wrapper_CUDA_cat)
```
Try to specify a single cuda device with `CUDA_VISIBLE_DEVICES=1` (or # of your GPU) before running the script. or set the device via `--device-map cuda:0` (or `--device cuda:0` in the alt image!) on the command line.

2. 4bit/8bit quantization and flash attention 2 don't work for all the models. No workaround, see: `sample.env` for known working configurations.

3. The default `--device-map auto` doesn't always work well with these models. If you have issues with multiple GPU's, try using `sequential` and selecting the order of your CUDA devices, like so:

```shell
# Example for reversing the order of 2 devices.
CUDA_VISIBLE_DEVICES=1,0 python vision.py -m llava-hf/llava-v1.6-34b-hf --device-map sequential
```

You can also use the environment variable: `OPENEDAI_DEVICE_MAP="sequential"` to specify the `--device-map` argument.

4. "My Nvidia GPU isn't detected when using docker."
- On Linux, you may need to specify the default runtime for your container environment (and perhaps install the nvidia-container-runtime), like so:
In /etc/docker/daemon.json:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```
- In Windows, be sure you have WSL2 installed and docker is configured to use it. Also make sure your nvidia drivers are up to date.

