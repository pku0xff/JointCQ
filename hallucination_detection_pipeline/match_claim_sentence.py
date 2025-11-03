import argparse
import json
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict, Any, Callable
from tqdm import trange
import warnings


def sentence_tokenize_process_dot(text, recover=False):
    if not recover:
        text = re.sub(r"O\.S\.B\.M. ", r"O.S.B.M.", text)
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2\3\4", text)
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Za-z])", r"\1\2\3", text)  # J. K. Campbell
        text = re.sub(r"((\n\s*)|(\. ))(\d+)\.\s+", r"\1\4.", text)  # 1. XXX
        text = re.sub(r"^(\d+)\.\s+", r"\1.", text)  # 1. XXX
        text = re.sub(r"(\W|^)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec|No|Op|D|Dr|St)\.\s+", r"\1\2.", text)
        text = re.sub(r"(\W|^)(et al)\.\s+([a-z])", r"\1\2.\3", text)
        text = re.sub(r"Alexander v\. Holmes", r"Alexander v.Holmes", text)
        text = re.sub(r"Brown v\. Board", r"Brown v.Board", text)
    else:
        text = re.sub(r"^(\d+)\.", r"\1. ", text)  # 1. XXX
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2 \3 \4", text)  # J. K. Campbell
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z][a-z])", r"\1\2 \3", text)  # J. Campbell
        text = re.sub(r"(\W|^)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec|No|Op|D|Dr|St)\.", r"\1\2. ", text)
        text = re.sub(r"(\W|^)(et al)\.([a-z])", r"\1\2. \3", text)

        text = re.sub("O\.S\.B\.M\.", "O.S.B.M. ", text)
        text = re.sub("U\. +S\.", "U.S.", text)
        text = re.sub("U\.S\. *S\. *R\.", "U.S.S.R.", text)
        text = re.sub("D\. +C\.", "D.C.", text)
        text = re.sub("D\. +Roosevelt", "D. Roosevelt", text)
        text = re.sub("A\. *D\. *(\W)", r"A.D.\1", text)
        text = re.sub("A\. +D\.", r"A.D.", text)
        text = re.sub("F\. +C\.", r"F.C.", text)
        text = re.sub("J\. +League", r"J.League", text)
        text = re.sub(r"Alexander v\. *Holmes", r"Alexander v. Holmes", text)
        text = re.sub(r"Brown v\. *Board", r"Brown v. Board", text)
    return text


def sentence_tokenize(text, language, keep_end, keep_colon=False):
    if language == 'zh':
        if not keep_colon:
            text = re.sub(r"([:：])(\s+)", r"\1", text)
        sents2 = []
        sents = re.split("(。|！|？|；|\n+)", text)
        # print(sents)
        for i in range(0, len(sents), 2):
            if i + 1 < len(sents):
                sent = sents[i] + sents[i + 1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sents2.append(sent)
        # print(sents2)
        return sents2
    elif language == 'en':
        text = sentence_tokenize_process_dot(text)
        if not keep_colon:
            text = re.sub(r"([:：])(\s+)", r"\1 ", text)

        sents2 = []
        sents = re.split("((?:[.!?;]\s+)|(?:\n+))", text)
        # print(sents)
        for i in range(0, len(sents), 2):
            if i + 1 < len(sents):
                sent = sents[i] + sents[i + 1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sent = sentence_tokenize_process_dot(sent, recover=True)
                sents2.append(sent)
        return sents2


def extract_label(annotation_str: str) -> str:
    """
    从 annotation 字符串中提取 <Hallucination> 与 <Reference> 之间的标签。
    若未找到则返回 'unknown'。
    """
    m = re.search(r"<Hallucination>\s*(.*?)\s*<Reference>", annotation_str, flags=re.S)
    if m is None:
        m = re.search(r"<幻觉>\s*(.*?)\s*<参考>", annotation_str, flags=re.S)
    label = m.group(1).strip() if m else 'unknown'
    if label.lower() == 'none' or label.lower() == '无':
        label = '正确'
    elif label.lower() == 'contradictory' or label.lower() == '矛盾':
        label = '幻觉'
    elif label.lower() == 'unverifiable' or label.lower() == '无法验证':
        label = '无法验证'
    else:
        label = 'unknown'
    return label


def preprocess_annotation(
        record: Dict,
) -> List[Dict[str, str]]:
    """
    将记录预处理成 [{'text': ..., 'label': ...}, ...] 格式。

    参数
    ----
    record : dict
        原始 JSON 记录（包含 'answer' 与 'annotation' 字段等）。
    sent_tokenize : callable
        句子分割函数，签名与 nltk.sent_tokenize 相同。

    返回
    ----
    List[dict]
        每个元素形如 {'text': <句子>, 'label': <标签>}
    """
    sentences = sentence_tokenize(record["answer"], language=record["language"], keep_end=False)
    annotations = record["annotation"]

    if not annotations:
        return []

    if len(sentences) != len(annotations):
        raise ValueError(
            f"Sentence / annotation length mismatch "
            f"({len(sentences)} sentences vs {len(annotations)} annotations)"
        )

    processed = []
    for sent, ann in zip(sentences, annotations):
        label = extract_label(ann)
        processed.append({"text": sent.strip(), "label": label})

    return processed


def match_by_similarity(
        method1_sentences: List[Dict[str, str]],
        method2_sentences: List[str],
        model,
        similarity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    # model = SentenceTransformer(model_name)
    device = model.device if hasattr(model, 'device') else torch.device("cpu")

    # ---------- method-1 is empty → unknown ----------
    if len(method1_sentences) == 0:
        warnings.warn("method1_sentences is empty")
        return [
            {
                "claim": z,
                "sentence": None,
                "assigned_label": "unknown",
                "similarity": 0.0
            }
            for z in method2_sentences
        ]

    # 提取文本与标签
    method1_texts = [s["text"] for s in method1_sentences]
    method1_labels = [s["label"] for s in method1_sentences]

    # 生成嵌入
    method1_embeddings = model.encode(method1_texts, convert_to_tensor=True).to(device)
    method2_embeddings = model.encode(method2_sentences, convert_to_tensor=True).to(device)

    results = []
    for idx2, emb2 in enumerate(method2_embeddings):
        cosine_scores = util.pytorch_cos_sim(emb2, method1_embeddings)[0]
        max_score, max_idx = torch.max(cosine_scores, dim=0)

        score_val = max_score.item()
        best_text = method1_texts[max_idx]
        best_label = method1_labels[max_idx]
        assigned = best_label if score_val >= similarity_threshold else "unknown"

        results.append({
            "claim": method2_sentences[idx2],
            "sentence": best_text,
            "assigned_label": assigned,
            "similarity": score_val
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    model_path = args.model_path
    output_dir = args.output_dir
    dataset = args.dataset

    file_path = os.path.join(output_dir, f'results_{dataset}.json')

    if 'anah' not in file_path:
        raise ValueError(f'{dataset} does not support sentence-level annotation')

    # load data
    with open(file_path) as f:
        data = json.load(f)

    model = SentenceTransformer(model_path)

    for i in trange(len(data)):
        records = preprocess_annotation(data[i])
        sent2label = {}
        for r in records:
            sent2label[r["text"]] = r["label"]
        data[i]['sent2label'] = sent2label
        for j, ceqg in enumerate(data[i]['claim_extraction']):
            claims = ceqg["claims"]
            claim_labels = match_by_similarity(records, claims, model, similarity_threshold=0.5)
            ceqg['claim_sentence_matching'] = claim_labels
            # ceqg['claim_labels'] = [i['assigned_label'] for i in claim_labels]
            data[i]['claim_extraction'][j] = ceqg

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
