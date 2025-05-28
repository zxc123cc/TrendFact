import json
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from fuzzywuzzy import fuzz


def parse(text):
    sentences = [
        "因此，该说法是正确的。",
        "因此，该说法是真实的。",
        "因此，该说法是错误的。",
        "因此，该说法是不真实的。",
        "因此，该说法是虚假的。",
        "因此，证据不足以验证该说法的真实性。"
    ]
    
    sentence_categories = [0, 0, 1, 1, 1, 2]

    sim_scores = []
    for sentence in sentences:
        similarity = fuzz.partial_ratio(sentence, text)
        sim_scores.append(similarity)

    max_score = max(sim_scores)
    max_index = sim_scores.index(max_score)

    return sentence_categories[max_index]


def compute_metrics_fn(preds, labels):
    assert len(preds) == len(labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro", labels=np.unique(labels))
    acc = accuracy_score(y_true=labels, y_pred=preds)
    p = precision_score(y_true=labels, y_pred=preds, average="macro", labels=np.unique(labels))
    r = recall_score(y_true=labels, y_pred=preds, average="macro", labels=np.unique(labels))
    return {
        "acc": acc,
        "macro_f1": f1,
        "macro_recall": r,
        "macro_precision": p
    }


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the JSON file containing prediction results."
    )
    args = parser.parse_args()

    now_path = args.input_file
    print(f"Loading data from {now_path}")
    
    with open(now_path, 'r') as f:
        pred_results = json.load(f)

    all_preds = []
    all_labels = []

    for sample in pred_results:
        if 'llm_response_parse' in sample:
            k = 'llm_response_parse'
        else:
            k = 'llm_response'
        if sample[k] == '':
            continue

        pred_label = parse(sample[k])
        all_preds.append(pred_label)
        all_labels.append(sample['label'])

    score_dic = compute_metrics_fn(all_preds, all_labels)
    print(f"acc = {score_dic['acc']},  macro_f1 = {score_dic['macro_f1']}, macro_recall = {score_dic['macro_recall']}, macro_precision = {score_dic['macro_precision']}")
