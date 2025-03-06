#!/bin/bash
DATE=$(date +%F-%H-%M)
FN="sample-${DATE}.env"
CSV="test_output-${DATE}.csv"
CSV_LOG="test_output-${DATE}.log"
touch ${FN} ${CSV}
docker run --runtime nvidia --gpus all \
  -v ./hf_home:/app/hf_home -v ./model_conf_tests.json:/app/model_conf_tests.json -v ./${FN}:/app/vision.sample.env -v ./${CSV}:/app/test_output.csv \
  -e HF_HOME=hf_home -e HF_HUB_ENABLE_HF_TRANSFER=1 -e HF_TOKEN=${HF_TOKEN} \
  -e CUDA_VISIBLE_DEVICES=1,0 -e OPENEDAI_DEVICE_MAP="sequential" \
  -e CLI_COMMAND="/usr/bin/env python test_models.py -v --log-level INFO" \
  -u $(id -u):$(id -g) --expose=5006 --name openedai-vision-test-${DATE} \
  ghcr.io/matatonic/openedai-vision 2> >(tee ${CSV_LOG} >&2)
