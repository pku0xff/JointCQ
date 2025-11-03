#!/bin/bash
set -e

output_dir="./output"
dataset="your_dataset"
pretrained_model_dir="your_model_dir"
serper_api_key=""

echo "Generating claims and queries"
CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_and_query.py --dataset $dataset --output_dir $output_dir --temperature 0.0 --num_return_seq 1 \
  --model_path "./model/claim_query_generator"  --tp 4

echo "Searching"
SERPER_API_KEY=$serper_api_key python search.py --output_dir $output_dir --dataset $dataset \
   --cache_file "serper_cache.jsonl"

echo "Verifying"
CUDA_VISIBLE_DEVICES=0,1,2,3 python detect.py --output_dir $output_dir --dataset $dataset --tp 4 \
   --model_path "$pretrained_model_dir/Qwen/Qwen3-14B"
