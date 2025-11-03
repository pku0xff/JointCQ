import argparse
import os
import random
import re
import json
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import jsonlines
import time
from tqdm import tqdm, trange

en_prompt_template = '''### Task
Given a claim and related reference searched by a query as input, your task is to determine whether the claim is valid based on the reference.

### Evaluation Criteria
Please make your judgment based on the following criteria and choose one of the three options:
1. Correct: The reference supports the claim.
2. Hallucination: The reference is relevant to the claim, but does not support the claim.
3. Irrelevant: The reference is irrelevant to the claim, thus does not contain enough information to determine the factuality of the claim. Only use this option when absolutely necessary.

Provide only one option as the output. No additional explanation is allowed.

### Input
[Claim]
{claim}
[Reference]
{reference}'''

zh_prompt_template = '''### 任务
给定一条陈述以及由查询检索到相关的参考资料作为输入，你的任务是根据参考资料判断陈述是否成立。

### 判断标准
请依据以下标准进行判断，输出三个选项之一：
1. 正确：参考资料能够支持陈述。
2. 幻觉：参考资料与陈述相关，但并不支持陈述。
3. 无关：参考资料与陈述内容无关，信息不足，无法判断陈述的真实性。非必要不使用此选项。

只输出一个选项，不允许有任何额外解释。

### 输入
[陈述]
{claim}
[参考资料]
{reference}'''

prompt_template = {
    'zh': zh_prompt_template,
    'en': en_prompt_template,
}


def get_ref(language, query, search_results):
    if language == 'zh':
        ref = '查询: '
    else:
        ref = 'Query: '
    ref += query + '\n'
    flatten = []
    for group in search_results:
        group = [i for i in group if i['source'] != 'Error']
        flatten.extend(group)
        # flatten.extend(group[:5])
    for i, item in enumerate(flatten):
        ref += f'{i + 1}. {item["content"]}\n'
    return ref.strip()


def load_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    detection_prompts = []
    for i in range(len(data)):
        for j in range(len(data[i]['claim_extraction'])):
            claims = data[i]['claim_extraction'][j]['claims']
            queries = data[i]['claim_extraction'][j]['queries']
            search_results = data[i]['claim_extraction'][j]['search_results']
            for k in range(len(claims)):
                if queries[k] == '':
                    continue
                curr_search = [search_results[k]]
                ref_text = get_ref(data[i]['language'], queries[k], curr_search)
                detection_prompts.append({
                    'question': data[i]['question'],
                    'claim': claims[k],
                    'query': queries[k],
                    'prompt': prompt_template[data[i]['language']].format(claim=claims[k], reference=ref_text),
                })
    # print(random.choice(detection_prompts)['prompt'])
    return detection_prompts


def parse_pred_texts(texts):
    value_count = {
        '幻觉': 0,
        '正确': 0,
        '无法验证': 0,
        'unknown': 0
    }
    reasoning = texts[0].split('</think>')[0].replace('<think>', '').strip()
    for t in texts:
        t = t.split('</think>')[-1].lower()
        if '正确' in t or 'correct' in t:
            value_count['正确'] += 1
        elif '幻觉' in t or 'hallucination' in t:
            value_count['幻觉'] += 1
        elif '无关' in t or 'irrelevant' in t:
            value_count['无法验证'] += 1
        else:
            value_count['unknown'] += 1
    pred = max(value_count, key=value_count.get)
    return {
        'pred': pred,
        'reasoning': reasoning,
    }


def postprocess(in_file, detection_results, out_file):
    with open(in_file) as f:
        data = json.load(f)
    qcq2detection_results = {}
    for item in detection_results:
        qcq2detection_results[item['question'] + item['claim'] + item['query']] = parse_pred_texts(item['pred_text'])
    for i in range(len(data)):
        for j in range(len(data[i]['claim_extraction'])):
            claims = data[i]['claim_extraction'][j]['claims']
            queries = data[i]['claim_extraction'][j]['queries']
            claim_predictions = []
            claim_reasonings = []
            for k in range(len(claims)):
                if queries[k] == '':
                    claim_predictions.append('无法验证')
                    claim_reasonings.append('')
                else:
                    result = qcq2detection_results[data[i]['question'] + claims[k] + queries[k]]
                    claim_predictions.append(result['pred'])
                    claim_reasonings.append(result['reasoning'])
            data[i]['claim_extraction'][j]['claim_predictions'] = claim_predictions
            data[i]['claim_extraction'][j]['claim_reasonings'] = claim_reasonings
    with open(out_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run(model, data, params):
    prompts = []
    for item in data:
        prompt = item['prompt']
        assert type(prompt) == str
        prompt = prompt[:5000]
        prompts.append([
            # {'role': 'system', 'content': "You are a helpful, respectful and honest assistant."},
            {"role": "user", "content": prompt}
        ])

    outputs = model.chat(prompts, params)

    for i in range(len(data)):
        data[i]['pred_text'] = [o.text for o in outputs[i].outputs]

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_return_seq', type=int, default=1)
    args = parser.parse_args()
    output_dir = args.output_dir
    dataset = args.dataset
    model_path = args.model_path
    tp = args.tp
    temperature = args.temperature
    num_return_seq = args.num_return_seq

    file_path = os.path.join(output_dir, f'search_{dataset}.json')

    data = load_data(file_path)

    model = LLM(model_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=0.9,
                max_model_len=8192)

    params = SamplingParams(temperature=temperature, n=num_return_seq, max_tokens=4096)

    data = run(model, data, params)

    with open(os.path.join(output_dir, f'judge_{dataset}.json'), 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, f'judge_{dataset}.json')) as f:
        data = json.load(f)

    postprocess(file_path, data, os.path.join(output_dir, f'results_{dataset}.json'))
