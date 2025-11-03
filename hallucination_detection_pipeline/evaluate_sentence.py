import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from evaluate_response import postprocess, f1_per_class


def text2label(text):
    t = text.lower()
    if '正确' in t or 'correct' in t:
        return '正确'
    elif '冲突' in t or 'contradict' in t or '幻觉' in t or 'hallucination' in t:
        return '幻觉'
    elif '无法验证' in t or 'unverifiable' in t or '无关' in t or 'irrelevant' in t:
        return '无法验证'
    else:
        return f'failed: {t}'


def process_unverifiable(labels, preds, mode='remove'):
    assert mode in ['remove', 'correct']
    processed_labels = []
    processed_preds = []
    for i in range(len(labels)):
        label = labels[i]
        pred = preds[i]
        if label == '无法验证' or label == 'unknown':
            continue
        if pred == '无法验证':
            if mode == 'remove':
                continue
            elif mode == 'correct':
                pred = '正确'
        processed_labels.append(label)
        processed_preds.append(pred)

    assert set(processed_labels) == set(processed_preds)
    assert set(processed_labels) == {'幻觉', '正确'}

    return processed_labels, processed_preds


def evaluate_sentence_level(file_path, mode='remove'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    data = postprocess(data)

    labels = defaultdict(list)  # {"en": [N_en], "zh": [N_zh]}, 1_en=[ce1, ce2, ce3], ce1=['正确', ...]
    predictions = defaultdict(list)
    for i, item in enumerate(data):
        language = item['language']
        sentence_labels = item['sent2label']
        for j, claim_extraction in enumerate(item['claim_extraction']):
            claim2predictions = {}
            for k in range(len(claim_extraction['claims'])):
                claim2predictions[claim_extraction['claims'][k]] = claim_extraction['claim_predictions'][k]
            sentence2claim_preds = {s: [] for s in sentence_labels.keys()}
            for assign in claim_extraction['claim_sentence_matching']:
                if assign['sentence'] is None:
                    continue
                sentence2claim_preds[assign['sentence']].append(claim2predictions[assign['claim']])
            sent2pred = {}
            for s, l in sentence_labels.items():
                labels[language].append(l)
                p = '无法验证'
                if '幻觉' in sentence2claim_preds[s]:
                    p = '幻觉'
                elif '正确' in sentence2claim_preds[s]:
                    p = '正确'
                sent2pred[s] = p
                predictions[language].append(p)
            data[i]['claim_extraction'][j]['sent2pred'] = sent2pred

    for k in labels.keys():
        l, p = process_unverifiable(labels[k], predictions[k], mode=mode)
        labels[k] = l
        predictions[k] = p

    def safe_get_f1(label_list, pred_list, target_label='幻觉', digits=4):
        print('N', len(label_list))
        f1_scores = f1_per_class(label_list, pred_list)
        return round(f1_scores.get(target_label, 0.0), digits)

    def evaluate_and_report(label_name, labels, predictions, report_text, target_label='幻觉'):
        acc = accuracy_score(labels, predictions)
        f1 = safe_get_f1(labels, predictions, target_label)
        report = classification_report(labels, predictions, digits=4)

        print(f"{label_name} acc: {acc:.4f}")
        print(f"{label_name} f1 ({target_label}): {f1:.4f}")

        report_text += f"{label_name} results:\n{report}\n\n"
        return report_text

    report_text = ""

    if 'en' in labels:
        report_text = evaluate_and_report("English", labels['en'], predictions['en'], report_text)

    if 'zh' in labels:
        report_text = evaluate_and_report("Chinese", labels['zh'], predictions['zh'], report_text)

    if 'en' in labels and 'zh' in labels:
        all_labels = labels['en'] + labels['zh']
        all_preds = predictions['en'] + predictions['zh']
        report_text = evaluate_and_report("All", all_labels, all_preds, report_text)

    data_dir = os.path.dirname(file_path)
    data_file = f'evaluated_sent_' + file_path.split('/')[-1]
    with open(os.path.join(data_dir, data_file), 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    data_file = f'metrics_sent_{mode}_' + file_path.split('/')[-1].replace('.json', '.txt')
    with open(os.path.join(data_dir, data_file), 'w') as f:
        f.write(report_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    output_dir = args.output_dir
    dataset = args.dataset

    print('Remove unverifiable predictions')
    evaluate_sentence_level(os.path.join(output_dir, f'results_{dataset}.json'), mode='remove')
    print('****************************************')
    print('Take unverifiable predictions as no hallucination')
    evaluate_sentence_level(os.path.join(output_dir, f'results_{dataset}.json'), mode='correct')
