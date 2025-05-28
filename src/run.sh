#!/bin/bash
python FactISR.py \
  --trend_data_path ../data/TrendFact.json \
  --retrieval_evidence_path ../outputs/retrieval_evidence_bge_m3.json \
  --output_file ../outputs/results.json \
  --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/linzhimin/comps/pretrained_model/qwen2.5/Qwen/QwQ-32B \
  --tensor_parallel_size 2 \
  --max_model_len 30000 \
  --max_tokens 8000 \
  --temperature 0.0 \
  --max_trunc_len 3000 \
  --max_evidence_limit 3 \
  --max_turn 5
