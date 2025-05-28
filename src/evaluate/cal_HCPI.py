import json
import argparse
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np

def get_data(input_path):
    with open(input_path, 'r') as f:
        pred_results = json.load(f)

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

        sim_scores = [fuzz.partial_ratio(sentence, text) for sentence in sentences]
        max_score = max(sim_scores)
        max_index = sim_scores.index(max_score)

        return sentence_categories[max_index]

    datas = []
    for sample in pred_results:
        if sample['ECS'] == '':
            continue
        k = 'llm_response' if 'llm_response_parse' not in sample else 'llm_response_parse'
        now_sample = {
            'influence_score': sample['influence_score'],
            'views': sample['views'],
            'Discussion': sample['Discussion'],
            'Engagemen': sample['Engagemen'],
            'Post': sample['Post'],
            'ECS': sample['ECS'],
            'label': sample['label'],
            'pred_label': parse(sample[k])
        }
        datas.append(now_sample)

    return datas

def process_special_units(df):
    df = df.copy()

    def convert_units(value):
        if pd.isna(value) or not isinstance(value, str):
            return value
        try:
            if '亿' in value:
                return float(value.replace('亿', '')) * 1e8
            elif '万' in value:
                return float(value.replace('万', '')) * 1e4
            return float(value)
        except:
            return value

    unit_cols = ['views', 'Discussion', 'Engagemen', 'Post']
    for col in unit_cols:
        df[col] = df[col].apply(convert_units)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in unit_cols:
        q25 = df[col].quantile(0.25)
        df[col] = df[col].fillna(q25)

    return df

def cal_HCPI(df):
    weights = {
        'views': 0.05,
        'Discussion': 0.2,
        'Engagemen': 0.15,
        'Post': 0.6
    }
    metrics = ['views', 'Discussion', 'Engagemen', 'Post']

    for col in metrics:
        historical_min = df[col][df[col] > 0].min()
        safe_min = max(historical_min * 0.1, 1)
        df[col] = df[col].fillna(historical_min).clip(lower=safe_min)

    def influence_scale_control(scores, max_ratio=10):
        min_score = scores.min()
        max_score = scores.max()
        current_ratio = max_score / min_score

        if current_ratio <= max_ratio:
            return scores

        target_max = min_score * max_ratio
        compression_factor = np.log(target_max / min_score) / np.log(current_ratio)
        scaled_scores = min_score * (scores / min_score) ** compression_factor

        return scaled_scores

    for col in metrics:
        df[f'log_{col}'] = np.log1p(df[col])

    EPS = 1e-8
    for col in metrics:
        log_col = f'log_{col}'
        q10 = df[log_col].quantile(0.10)
        q90 = df[log_col].quantile(0.90)
        df[f'{col}_norm'] = (df[log_col] - q10) / (q90 - q10 + EPS)
        df[f'{col}_norm'] = 0.2 + 0.6 / (1 + np.exp(-df[f'{col}_norm']))

    df['base_influence'] = df.apply(
        lambda row: sum(weights[col] * row[f'{col}_norm'] for col in metrics), axis=1)

    df['final_influence'] = df['influence_score'] * df['base_influence']
    df['final_influence'] = influence_scale_control(df['final_influence'], max_ratio=10)

    sum_score = 0
    now_score = 0
    for _, row in df.iterrows():
        sum_score += row['final_influence']
        if row['label'] == 0:
            if row['pred_label'] == 0:
                now_score += row['final_influence'] * (float(row['ECS']) * 0.2)
            elif row['pred_label'] == 1:
                now_score -= 2 * row['final_influence']
        elif row['label'] == 1:
            if row['pred_label'] == 1:
                now_score += row['final_influence'] * (float(row['ECS']) * 0.2)
        else:
            if row['pred_label'] == 1:
                now_score -= row['final_influence']
            if row['pred_label'] == 2:
                now_score += row['final_influence'] * (float(row['ECS']) * 0.2)

    print(now_score / sum_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input file path.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file."
    )
    args = parser.parse_args()

    df = pd.DataFrame(get_data(args.input_file))
    df = process_special_units(df)
    cal_HCPI(df)
