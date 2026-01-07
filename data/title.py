import csv
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_path', type=str, default='./QG/total_data.csv')
parser.add_argument('--output_path', type=str, default='./id_title.txt')
parser.add_argument('--model_name_or_path', type=str, default='/home/aiphys/suzitao/models/mt5-base')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

with open(args.corpus_path, 'r', encoding='gb18030', errors='ignore', newline='') as fr, \
     open(args.output_path, 'w', encoding='utf-8') as fw:

    reader = csv.DictReader(fr)  # 字段：id,location,content
    for row in tqdm(reader):
        old_id = row['id']
        location = row['location']
        content = row['content']

        text = f"{location}"  # 或者只用 content，看你需求
        input_ids = tokenizer(text, add_special_tokens=False).input_ids
        new_id = ",".join(str(x) for x in input_ids) + ",1"

        fw.write(str(old_id) + '\t' + new_id + '\n')
