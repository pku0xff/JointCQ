import argparse
import os
import random
import re
import json
from collections import defaultdict
from vllm import LLM, SamplingParams

import jsonlines
import time
from tqdm import tqdm, trange

prompt_template = {
    'zh': '''### 任务
给定问题和回答作为输入，你的任务是提取所有的陈述，然后为每条陈述生成一条搜索引擎查询，用于协助对陈述进行事实核查。

### 任务规则
提取陈述时请严格遵循以下规则：
1. 陈述必须是可以核实或驳斥的事实性声明。排除主观意见、情绪表达和模糊判断。
2. 每条陈述必须语义完整，不依赖上下文即可独立理解其含义。
3. 陈述中禁止使用指代不明的代词，必须使用具体名词表述。
4. 必须提取并输出所有符合条件的陈述，每条陈述独占一行。
5. 当输入中不存在符合上述标准的陈述时，输出"无陈述"。

生成查询时请严格遵循以下规则：
1. 查询应当简洁明确，对待验证的陈述具有针对性。
2. 查询能够应用于搜索引擎的搜索，帮助用户获取有效信息。
3. 若无待查询的内容，直接输出“无查询”。

### 输入
[问题]
{question}
[回答]
{answer}''',
    'en': '''### Task
Given a question and an answer as input, your task is to extract all claims, and generate a search engine query for each claim to help fact-check the claims.

### Task Rules
When extracting claims, strictly follow these rules:
1. Claims must be factual statements that can be verified or refuted. Exclude subjective opinions, emotional expressions, and vague judgments.
2. Each claim must be semantically complete and independently understandable without relying on context.
3. Avoid ambiguous pronouns in claims. Use specific nouns for clarity.
4. Extract and output all qualifying claims, with each claim on a separate line.
5. If no claims meeting the above criteria exist in the input, output "No claims."

When generating the queries, strictly follow these rules:
1. The queries should be concise and clear, specifically targeting the claims to be verified.
2. The queries should be applicable to search engines and can help users obtain valid information.
3. If there is nothing to query, output "No query".

### Input
[Question]  
{question}  
[Answer]  
{answer}'''
}


def get_prompt(item):
    return prompt_template[item['language']].format(question=item['question'], answer=item['answer'])


def load_data(data_path):
    with open(data_path) as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i]['extraction_prompt'] = get_prompt(data[i])
    # print(random.choice(data)['extraction_prompt'])
    return data


def postprocess(data):
    for i in range(len(data)):
        extractions = []
        for output_text in data[i]['claim_output']:
            output_text = output_text.strip()
            if data[i]['language'] == 'zh':
                claim_text = output_text.split('[查询]')[0]
                claim_text = claim_text.split('[陈述]')[-1].strip()
                query_text = output_text.split('[结束]')[0]
                query_text = query_text.split('[查询]')[-1].strip()
            else:
                claim_text = output_text.split('[Queries]')[0]
                claim_text = claim_text.split('[Claims]')[-1].strip()
                query_text = output_text.split('[End]')[0]
                query_text = query_text.split('[Queries]')[-1].strip()
            claims = claim_text.split('\n')
            for j, c in enumerate(claims):
                prefix = str(j + 1) + ' '
                c = c.strip()[len(prefix):].strip()
                claims[j] = c
            queries = query_text.split('\n')
            for j, q in enumerate(queries):
                prefix = str(j + 1) + ' '
                q = q.strip()[len(prefix):].strip()
                queries[j] = q
            # assert len(claims) == len(queries)
            if len(claims) != len(queries):
                # print(len(claims), len(queries))
                # print(claims)
                # print(queries)
                n = min(len(claims), len(queries))
                claims = claims[:n]
                queries = queries[:n]
            extractions.append({
                'claims': claims,
                'queries': queries,
            })
        data[i]['claim_extraction'] = extractions
    return data


def run(model, data, params):
    prompts = []
    for item in data:
        prompt = item['extraction_prompt']
        prompts.append([
            {'role': 'system', 'content': "You are a helpful, respectful and honest assistant."},
            {"role": "user", "content": prompt}
        ])

    outputs = model.chat(prompts, params)

    for i in range(len(data)):
        data[i]['claim_output'] = [o.text for o in outputs[i].outputs]

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tp', type=int)
    parser.add_argument('--output_dir', type=str, default='./output')
    # generation config
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_return_seq', type=int, default=1)
    args = parser.parse_args()
    dataset = args.dataset
    model_path = args.model_path
    tp = args.tp
    output_dir = args.output_dir
    temperature = args.temperature
    num_return_seq = args.num_return_seq

    os.makedirs(args.output_dir, exist_ok=True)
    # 1. load data
    data = load_data(f'./data/{dataset}.json')

    # 2. load model
    model = LLM(model_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=0.9,
                max_model_len=8192)

    params = SamplingParams(temperature=temperature, n=num_return_seq, max_tokens=4096)

    # 3. running
    data = run(model, data, params)

    data = postprocess(data)
    with open(os.path.join(output_dir, f'ceqg_{dataset}.json'), 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
