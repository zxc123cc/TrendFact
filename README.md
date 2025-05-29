# TrendFact: A Benchmark for Explainable Hotspot Perception in Fact-Checking with Natural Language Explanation

## Paper: [arxiv](https://arxiv.org/abs/2410.15135)
![Alt text](figures/TrendFact.png)

## Leaderboard



### Evidence Retrieval

| Method           | R@1       | R@2       | R@3       | R@5       |
|------------------|-----------|-----------|-----------|-----------|
| BM25 -w/o date   | 35.92     | 48.11     | 56.54     | 65.94     |
| BM25             | 36.70     | 49.07     | 57.28     | 66.75     |
| text-emb-ada-002 | 28.35     | 37.42     | 42.56     | 61.93     |
| bge-m3(dense)    | **49.02** | **61.48** | **69.67** | **78.97** |

<br>

### Fact Verification

| Method                 | Accuracy  | F1-Score  | Precision | Recall    |
|------------------------|-----------|-----------|-----------|-----------|
| PROGRAM-FC             | 45.24     | 44.28     | 43.34     | 45.30     |
| CLAIMDECOMP            | 47.48     | 46.40     | 45.32     | 47.53     |
| QwQ-32B-Preview        | 56.73     | 53.30     | 55.92     | 57.27     |
| Qwen2.5-72B-Instruct   | 58.64     | 51.96     | 58.90     | 56.28     |
| Qwen3-32B(No Think)    | 61.51     | 55.01     | 58.11     | 58.07     |
| DeepSeek-V3            | 63.42     | 57.17     | 60.35     | 60.44     |
| GPT-4o                 | 62.29     | 59.45     | 60.54     | 63.05     |
| Qwen3-32B(Think)       | 70.09     | 65.20     | 64.97     | 67.17     |
| DeepSeek-R1            | 71.67     | 64.94     | 65.31     | 66.00     |
| QwQ-32B                | 72.67     | 66.68     | 66.96     | 68.00     |
| **FactISR(QwQ-32B)**   | **74.32** | **67.58** | **69.08** | **68.06** |
| **FactISR(Qwen3-32B)** | **72.46** | **65.01** | **66.07** | **65.29** |

<br>

### Explanation Generation

| Method               | HCPI       | ECS        | BLEU-4     | BERTScore  | ROUGE-1    | ROUGE-2    | ROUGE-L    |
|----------------------|------------|------------|------------|------------|------------|------------|------------|
| QwQ-32B-Preview      | 0.4923     | 0.7689     | 0.1474     | 0.7479     | 0.4544     | 0.2702     | 0.3928     |
| Qwen2.5-72B-Instruct | 0.5321     | 0.7193     | **0.2994** | 0.8163     | **0.6000** | **0.4128** | **0.5446** |
| Qwen3-32B(No Think)  | 0.5172     | 0.7587     | 0.2740     | **0.8166** | 0.5921     | 0.3949     | 0.5389     |
| DeepSeek-V3          | 0.5718     | 0.7623     | 0.2609     | 0.8058     | 0.5711     | 0.3705     | 0.5182     |
| GPT-4o               | 0.5655     | 0.7972     | 0.2351     | 0.7934     | 0.5456     | 0.3380     | 0.4794     |
| Qwen3-32B(Think)     | 0.5679     | 0.8279     | 0.2378     | 0.7962     | 0.5475     | 0.3402     | 0.4897     |
| DeepSeek-R1          | 0.6032     | **0.8430** | 0.2144     | 0.7833     | 0.5152     | 0.3111     | 0.4519     |
| QwQ-32B              | 0.6110     | 0.8355     | 0.2214     | 0.7858     | 0.5237     | 0.3163     | 0.4622     |
| FactISR(QwQ-32B)     | **0.6336** | 0.8375     | 0.2185     | 0.7866     | 0.5251     | 0.3198     | 0.4645     |
| FactISR(Qwen3-32B)   | 0.6157     | 0.8268     | 0.2443     | 0.8015     | 0.5604     | 0.3585     | 0.5097     |


## Setup

1. Clone this repository:
```bash
git clone https://github.com/zxc123cc/TrendFact.git
cd TrendFact
```

2. Install required packages:
```bash
pip install -r requirements.txt
```


## Run
```bash
cd src
```

```bash
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
```
or
```bash
sh run.sh
```


## Evaluation

### Acc&F1
```bash
python evaluate/cal_Acc_F1.py --input_file ../outputs/results.json
```


### Text Quality
```bash
python evaluate/cal_text_quality.py --input_file ../outputs/results.json
```

### ECS
```bash
python evaluate/cal_ECS.py --input_file ../outputs/results.json --output_file ../outputs/results_ECS.json
```

### HCPI
```bash
python evaluate/cal_HCPI.py --input_file ../outputs/results_ECS.json
```

## Citation
```bibtex
@article{zhang2025trendfact,
  title={TrendFact: A Benchmark for Explainable Hotspot Perception in Fact-Checking with Natural Language Explanation},
  author={Zhang, Xiaocheng and Wang, Xi and Lu, Yifei and Wang, Jianing and Ye, Zhuangzhuang and Bao, Mengjiao and Yan, Peng and Su, Xiaohong},
  journal={arXiv preprint arXiv:2410.15135},
  year={2025},
  url={https://arxiv.org/abs/2410.15135}
}
```

