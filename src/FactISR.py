import os
import json
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict
from vllm import LLM, SamplingParams
import re
import random
import argparse
from prompts import factisr_system_prompt_qwq32b

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_truncation_text(tokenizer, text, tranc_len=200):
    text_ids = tokenizer(
        [text], add_special_tokens=False,
        return_tensors='pt')['input_ids'][0][:tranc_len]
    text = tokenizer.decode(text_ids)
    return text

def get_user_message(sample, evidence):
    user_message = '给定当前的说法和证据如下：\n'
    user_message += f'说法:{sample["claim"]}\n'
    user_message += f'初始证据:【{evidence[0]}】\n'
    user_message += f'请根据上述信息，按要求判断当前说法的真实性并给出一段文本解释。'
    return user_message


def create_isr_processor(tokenizer, delta=2.0, decay_factor=0.5, sca=10):
    negative_mid_words = {'否': '是', '不': '是'}
    nwid2pwid = {}
    for k, v in negative_mid_words.items():
        nwid = tokenizer.encode(k, add_special_tokens=False)[-1]
        pwid = tokenizer.encode(v, add_special_tokens=False)[-1]
        nwid2pwid[nwid] = pwid

    ending_id = tokenizer.eos_token_id

    target_phrase = "审视结果为："
    target_ids = tokenizer.encode(target_phrase, add_special_tokens=False)

    if not target_ids:
        raise ValueError(
            f"Target phrase '{target_phrase}' was encoded into an empty list. Please check the tokenizer or phrase.")

    def logits_processor(input_ids, scores):
        nonlocal sca

        if isinstance(input_ids, tuple):
            if len(input_ids) == 0:
                return scores
            input_ids = input_ids[0]

        if not isinstance(input_ids, torch.Tensor):
            return scores

        input_ids = input_ids.squeeze()

        if len(input_ids) >= len(target_ids):
            recent_tokens = input_ids[-len(target_ids):].tolist()
            if recent_tokens == target_ids:
                # 应用奖励逻辑
                top_ids = torch.topk(scores, 1)[1].item()
                for nwid, pwid in nwid2pwid.items():
                    if nwid == top_ids:
                        scores[pwid] += delta * sca
                sca *= decay_factor  # 衰减

        if ending_id in input_ids.tolist():
            sca = 10

        return scores

    return logits_processor

# Initialize the LLM
def init_llm(args):
    llm = LLM(model=args.model_path,
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len)
    tokenizer = llm.get_tokenizer()

    return llm, tokenizer

def run_generation(args, llm, tokenizer, sequences: List[Dict]) -> List:
    prompts = [s['prompt'] for s in sequences]
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=[BEGIN_ADD_EVIDENCE, tokenizer.eos_token],
        include_stop_str_in_output=True,
        # logits_processors=[create_isr_processor(tokenizer)] //todo: adapt on vllm
    )
    output_list = llm.generate(prompts, sampling_params=sampling_params)
    return output_list


def remove_evidence_sections(text):
    pattern = r'<\|begin_add_evidence\|>.*?<\|end_add_evidence\|>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def parse(text):
    try:
        text = text.split('</think>')[1]
        text = text.replace('\n', '')
    except:
        text = text.replace('\n', '')
    text = remove_evidence_sections(text)
    return text




BEGIN_ADD_EVIDENCE = "<|begin_add_evidence|>"
END_ADD_EVIDENCE = "<|end_add_evidence|>"
BEGIN_THINK = "<think>"
END_THINK = "</think>"




def get_datas(args, tokenizer):
    with open(args.trend_data_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    with open(args.retrieval_evidence_path, 'r', encoding='utf-8') as f:
        evidence_list = json.load(f)
        new_evidence_list = []
        for evidence in evidence_list:
            now_evidence = [get_truncation_text(tokenizer, e, args.max_trunc_len) for i, e in enumerate(evidence[:3])]
            new_evidence_list.append(now_evidence)
        evidence_list = new_evidence_list

    input_list = []
    for sample, evidence in zip(datas, evidence_list):
        now_input = get_user_message(sample, evidence)
        messages = [
            {"role": "system", "content": factisr_system_prompt_qwq32b},
            {"role": "user", "content": now_input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Qwen3 Switches between thinking and non-thinking modes. Default is True.
        )
        input_list.append(text)

    active_sequences = [{
        'item': item,
        'evidence': evidence[1:3],
        'prompt': prompt,
        'output': '',
        'finished': False,
        'history': [],
        'evidence_count': 0
    } for item, evidence, prompt in zip(datas, evidence_list, input_list)]

    return active_sequences



def run(args):
    turn = 0
    llm, tokenizer = init_llm(args)
    active_sequences = get_datas(args, tokenizer)

    while True:
        sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]

        if sequences_needing_generation:
            turn += 1
            print(f'\n-------------- Turn {turn} --------------')
            print(f"{len(sequences_needing_generation)}条样本正在执行...")
            outputs = run_generation(args, llm, tokenizer, sequences_needing_generation)

            batch_sequences = []

            for seq, out in zip(sequences_needing_generation, outputs):
                text = out.outputs[0].text
                seq['history'].append(text)
                seq['prompt'] += text
                seq['output'] += text

                if seq['output'].rstrip().endswith(BEGIN_ADD_EVIDENCE):
                    if seq['evidence_count'] < args.max_evidence_limit:
                        now_evidence = seq['evidence'][seq['evidence_count']]
                        batch_sequences.append(seq)

                        seq['evidence_count'] += 1

                        append_text = f"\n{now_evidence}{END_ADD_EVIDENCE}\n\n好的，我现在拿到了一条新增的证据，我将结合先前内容判断新增证据能否验证当前说法的真实性"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)

                    elif seq['evidence_count'] >= args.max_evidence_limit:
                        limit_message = f"\n超过了最大证据数量，你不能再额外获得证据\n{END_ADD_EVIDENCE}\n\n好的，系统返回已超过最大证据数量，目前仍然判断不了说法的真实性，因此可以将说法的标签定义为“证据不充分”，并按要求进行解释生成。"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                else:
                    seq['finished'] = True

        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            break
        else:
            if turn >= args.max_turn:
                print(f"达到最大轮数{args.max_turn}, 退出.")
                break

    final_data = []
    for data in active_sequences[:]:
        if not data['finished']:
            continue
        sample = data['item']
        sample['llm_response'] = data['output']
        sample['llm_response_parse'] = parse(data['output'])
        final_data.append(sample)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_data, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LLM processing on specified data.")
    parser.add_argument("--trend_data_path", type=str, required=True, help="Path to the input trend data JSON file.")
    parser.add_argument("--retrieval_evidence_path", type=str, required=True, help="Path to the retrieval evidence JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM model directory.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for the model.")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum model length.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens for generation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation.")
    parser.add_argument("--max_trunc_len", type=int, default=200, help="Maximum truncation length for text.")
    parser.add_argument("--max_evidence_limit", type=int, default=3, help="Maximum number of evidences.")
    parser.add_argument("--max_turn", type=int, default=10, help="Maximum number of processing turns.")
    args = parser.parse_args()
    args.max_evidence_limit -= 1

    setup_seed(42)
    run(args)

