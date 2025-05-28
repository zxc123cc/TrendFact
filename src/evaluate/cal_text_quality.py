import json
import argparse
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BertTokenizer, BertModel
from bert_score import score
from rouge import Rouge
import torch
import bert_score


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def calculate_bert_score(sent1, sent2, model_path='bert-base-chinese'):
    max_length = 512
    inputs_sent1 = tokenizer(sent1, truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')
    inputs_sent2 = tokenizer(sent2, truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')
    processed_sent1 = tokenizer.decode(inputs_sent1['input_ids'][0], skip_special_tokens=True)
    processed_sent2 = tokenizer.decode(inputs_sent2['input_ids'][0], skip_special_tokens=True)
    P, R, F1 = bert_score.score([processed_sent1], [processed_sent2], model_type=model_path, lang="zh", verbose=True)
    return F1.item()

def cal_one_sample_score(hypothesis, reference, bert_model_type):
    hypothesis_tokens = list(jieba.cut(hypothesis))
    reference_tokens = list(jieba.cut(reference))
    smooth_func = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smooth_func)
    bert_score = calculate_bert_score(hypothesis, reference)
    rouge = Rouge()
    rouge_scores = rouge.get_scores(" ".join(hypothesis_tokens), " ".join(reference_tokens))[0]
    return {
        "BLEU-4": bleu_score,
        "BERTScore": bert_score,
        "ROUGE-1": rouge_scores['rouge-1'],
        "ROUGE-2": rouge_scores['rouge-2'],
        "ROUGE-L": rouge_scores['rouge-l']
    }

def evaluate_metrics(hypothesis_list, reference_list, bert_model_type):
    metrics = {
        "BLEU-4": [],
        "BERTScore": [],
        "ROUGE-1": [],
        "ROUGE-2": [],
        "ROUGE-L": []
    }
    for hypothesis, reference in zip(hypothesis_list, reference_list):
        if hypothesis == '':
            continue
        try:
            now_metrics = cal_one_sample_score(hypothesis, reference, bert_model_type=bert_model_type)
            metrics['BLEU-4'].append(now_metrics['BLEU-4'])
            metrics['BERTScore'].append(now_metrics['BERTScore'])
            metrics['ROUGE-1'].append(now_metrics['ROUGE-1']['f'])
            metrics['ROUGE-2'].append(now_metrics['ROUGE-2']['f'])
            metrics['ROUGE-L'].append(now_metrics['ROUGE-L']['f'])
        except:
            continue
    
    metrics['BLEU-4'] = sum(metrics['BLEU-4']) / len(metrics['BLEU-4'])
    metrics['BERTScore'] = sum(metrics['BERTScore']) / len(metrics['BERTScore'])
    metrics['ROUGE-1'] = sum(metrics['ROUGE-1']) / len(metrics['ROUGE-1'])
    metrics['ROUGE-2'] = sum(metrics['ROUGE-2']) / len(metrics['ROUGE-2'])
    metrics['ROUGE-L'] = sum(metrics['ROUGE-L']) / len(metrics['ROUGE-L'])
    
    return metrics

def main(input_file):
    with open(input_file, 'r') as f:
        pred_results = json.load(f)

    pred_list = []
    eval_data = []
    for sample in pred_results:
        if 'llm_response_parse' in sample:
            k = 'llm_response_parse'
        else:
            k = 'llm_response'
        if sample[k] == '':
            continue
        pred_list.append(sample[k])
        eval_data.append(sample['explanation'])

    metrics = evaluate_metrics(pred_list, eval_data, bert_model_type='bert-base-chinese')
    print(input_file)
    print(metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model predictions using various metrics.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file."
    )
    args = parser.parse_args()
    
    main(args.input_file)
