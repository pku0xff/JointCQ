# README

### Configuration Requirements

1. Run `conda env create -f environment.yml` to build the conda environment.
2. Download pretrained models `Qwen/Qwen3-14B`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
3. Place the model `claim_query_generator` under dir `./model`.
4. For Serper API key, refer to https://serper.dev/.

### Implementation

1. **Detection**: `extract_and_query.py` is the implementation of **Claim-Query Genertor**. `search.py` is the implementation of **Searcher**. `detect.py` is the implementation of **Verifier**.

2. **Evaluation**: `evaluate_response.py` is for response-level evaluation. `evaluate_sentence.py` is for sentence-level evaluation, only supporting the ANAH dataset.

### 数据准备

The default dir for data is `./data`. `.json` format is required.

Example:

```
[
  {
    "question": "蜡梅为何得名？",
    "answer": "蜡梅得名是因为它与梅花同时开放，香气相似，颜色似蜜蜡，故得此名。",
    "language": "zh"
  },
  ...
]
```

### Scripts

Specify the arguments at the beginning of the scripts:
```
output_dir="./output"
dataset="your_dataset"
pretrained_model_dir="your_model_dir"
serper_api_key="your_api_key"
```

### Output Format

The results will be saved in './output/results_{your_dataset}.json'.

Example:
```
{
    "question": "中国第一部故事长篇电影是？",
    "answer": "中国第一部故事长篇电影应该是1921年上映的《神女》。",
    "language": "zh",
    ...
    "claim_extraction": [
      {
        "claims": [
          "中国第一部故事长篇电影是1921年上映的《神女》。"
        ],
        "queries": [
          "中国第一部故事长篇电影是哪部？"
        ],
        "search_results": [...],
        "claim_reasonings": [...],
        "claim_predictions": [
          "幻觉"
        ],
        "claim_modes": [...]
      }
    ]
},
```

Here elements in `claims`, `queries`, `search_results`, `claim_reasonings`, `claim_predictions` are corresponded by their order.