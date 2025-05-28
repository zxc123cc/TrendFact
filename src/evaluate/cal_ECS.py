import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process input and output file paths.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSON file."
    )
    return parser.parse_args()

def gen_result(system, message, temperature):
    url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
    messages = [
        system,
        {"role": "user", "content": message},
    ]
    api_key = "YOUR_API_KEY"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    model = 'gpt-4o-2024-11-20'
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    content = json.dumps(data)

    try:
        response = requests.post(url, headers=headers, data=content)
        reply = response.json()['choices'][0]['message']['content']
        return 1, reply
    except Exception as e:
        print(e)
        return 0, ""

def request_result(system, message, temperature):
    max_retry_nums = 3
    cnt = 0
    res = ""
    while cnt <= max_retry_nums:
        flag, res = gen_result(system, message, temperature)
        if flag == 1:
            break
        else:
            cnt += 1
    return res

def get_user_message(sample):
    user_message = '给定当前的说法和黄金解释如下：\n'
    user_message += f'说法:{sample["claim"]}\n'
    user_message += f'黄金解释1为：{sample["explanation"]}\n'
    if 'llm_think' in sample:
        user_message += f'生成解释2为：\n思考过程：{sample["llm_think"]}\n'
        user_message += f'最终输出：{sample["llm_response"]}\n'
    else:
        if 'llm_response_parse' in sample:
            user_message += f'生成解释2为：{sample["llm_response_parse"]}\n'
        else:
            user_message += f'生成解释2为：{sample["llm_response"]}\n'
    user_message += '请根据上述信息，生成一个一致性分数：'
    return user_message

def process_row(sample, system):
    if 'llm_response_parse' in sample and sample['llm_response_parse'] == '':
        sample['ECS'] = ""
    elif 'llm_response' in sample and sample['llm_response'] == '':
        sample['ECS'] = ""
    else:
        now_input = get_user_message(sample)
        response = request_result(system, now_input, 0.0)
        sample['ECS'] = response
    return sample

if __name__ == '__main__':
    args = parse_arguments()
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    system = {"role": "system",
              "content": '''
你是一个事实核查专家，事实核查分为事实验证和解释生成两个任务。
你可以将事实验证理解为判断一个说法是否是真实的，将解释生成理解为对这个说法的真实性作出解释，表明这个说法为什么是对的/错的/证据不充分的。
现在我提供给你当前说法，一个黄金解释1，以及一个生成解释2。
你的任务是判断黄金解释1和生成解释2之间的一致性，并将其量化为一个分数。
这个分数有5个等级，即你打分的范围在1～5。
接下来我将给你具体分数等级的含义：
1:真实性不一致，例如黄金解释1里有“该说法是正确的”，而生成解释2里是“该说法是错误的”，并且解释内容也完全不相关。
2:真实性不一致，但解释内容相关。
3:真实性一致，解释内容不相关。
4:真实性一致，解释内容部分相关。
5:真实性一致，解释内容完全相关。
请你根据当前说法，一个黄金解释1，以及一个生成解释2量化一个一致性分数。
注意：有的模型输出的结果可能包含其思考过程，这些思考过程中有关键信息和黄金解释相关，也应该考虑在内，但真实性只关注最终输出里的。
注意：这里的完全相关不是指文本内容完全一样，是指意思一样，文本内容很相近即可。
注意，你的输出只为一个阿拉伯数据，其范围为1～5。
'''}

    datas = []
    futures = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for sample in data:
            futures.append(executor.submit(process_row, sample, system))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            datas.append(result)
            if len(datas) % 5 == 0:
                with open(args.output_file, 'w') as f:
                    f.write(json.dumps(datas, ensure_ascii=False))

    print(len(datas))
    with open(args.output_file, 'w') as f:
        f.write(json.dumps(datas, ensure_ascii=False))


    score_list = []
    for sample in datas:
        if sample['ECS'] == '':
            continue
        try:
            score_list.append(0.2*int(sample['ECS']))
        except:
            continue
    print(sum(score_list) / len(score_list))
