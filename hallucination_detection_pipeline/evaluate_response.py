import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def text2label(text):
    t = text.lower()
    if '正确' in t or 'correct' in t:
        return '正确'
    elif '冲突' in t or 'contradict' in t or '幻觉' in t or 'hallucination' in t:
        return '幻觉'
    elif '无法验证' in t or 'unverifiable' in t or '无关' in t or 'irrelevant' in t:
        return '无法验证'
    else:
        print('failed:' + text)
        input()
        return f'failed: {t}'


def claim2response_pred(claim_predictions):
    claim_predictions = [p.lower() for p in claim_predictions if
                         p != '无法验证' and p.lower() != 'unverifiable' and p != 'unknown']
    if not claim_predictions:
        pred = '无法验证'
    elif '幻觉' in claim_predictions or 'hallucination' in claim_predictions:
        pred = '幻觉'
    else:
        pred = '正确'
    return pred


def postprocess(data):
    for i in range(len(data)):
        for j in range(len(data[i]['claim_extraction'])):
            #if 'claim_predictions' not in data[i]['claim_extraction'][j].keys():
            #    data[i]['claim_extraction'][j]['claim_predictions'] = data[i]['detect_outputs'][j].split('\n')
            claim_preds = [text2label(p) for p in data[i]['claim_extraction'][j]['claim_predictions']]
            data[i]['claim_extraction'][j]['claim_predictions'] = claim_preds
            data[i]['claim_extraction'][j]['response_prediction'] = claim2response_pred(claim_preds)
    return data


def process_unverifiable(labels, preds, mode='remove'):
    assert mode in ['remove', 'correct']
    assert len(labels) == len(preds)
    processed_labels = []
    processed_preds = []
    for i in range(len(labels)):
        label = labels[i]
        pred = preds[i]
        if label == '无法验证':
            continue
        if pred == '无法验证':
            if mode == 'remove':
                continue
            elif mode == 'correct':
                pred = '正确'
        processed_labels.append(label)
        processed_preds.append(pred)

    return processed_labels, processed_preds


def f1_per_class(y_true, y_pred):
    labels = np.unique(y_true)
    f1_vals = f1_score(y_true, y_pred, average=None, labels=labels)
    return {str(label): float(score) for label, score in zip(labels, f1_vals)}


def evaluate(file_path, mode='remove'):
    with open(file_path, 'r') as f:
        data = json.load(f)

    labels = defaultdict(list)
    predictions = defaultdict(list)

    if 'response_prediction' not in data[0]['claim_extraction'][0].keys():
        data = postprocess(data)
    unveri_cnt = 0
    fail_cnt = 0
    for item in data:
        if 'label' not in item.keys():
            fail_cnt += 1
            continue
        pred = item['claim_extraction'][0]['response_prediction']
        if item['label'] == 0:
            label = '正确'
        elif item['label'] == 1:
            label = '幻觉'
        else:
            unveri_cnt += 1
            label = '无法验证'
        predictions[item['language']].append(pred)
        labels[item['language']].append(label)

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
    data_file = f'evaluated_{mode}' + file_path.split('/')[-1]
    with open(os.path.join(data_dir, data_file), 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    data_file = f'metrics_{mode}' + file_path.split('/')[-1].replace('.json', '.txt')
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
    evaluate(os.path.join(output_dir, f'results_{dataset}.json'), mode='remove')
    print('****************************************')
    print('Take unverifiable predictions as no hallucination')
    evaluate(os.path.join(output_dir, f'results_{dataset}.json'), mode='correct')
