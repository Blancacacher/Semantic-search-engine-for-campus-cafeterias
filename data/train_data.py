import json
import random
import os
import re
import csv

# 固定随机种子，保证可复现
seed = 42
random.seed(seed)

# ======== 路径配置（按你现在的文件来） ========

base_dir = "/home/aiphys/suzitao/IR/data"

docid_path = os.path.join(base_dir, "id_title.txt")
corpus_path = os.path.join(base_dir, "QG/total_data.csv")
qg_path = os.path.join(base_dir, "QG/total_queries.jsonl")

output_dir = os.path.join(base_dir, "train_title")  # 你可以改名字
os.makedirs(output_dir, exist_ok=True)


# ======== 读取 id -> docid 映射 ========

def load_docid(docid_path):
    """
    读取 id_title.txt，假设每行: 内部id \t docid
    """
    id2docid = {}
    with open(docid_path, encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            _id, docid = parts[0], parts[1]
            id2docid[str(_id)] = docid
    return id2docid


# ======== 中文句子切分（保留你原来的逻辑） ========

def split_chinese_sentences(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'([。！？])', r'\1<SPLIT>', text)
    text = re.sub(r'([^\n。！？])\n', r'\1<SPLIT>', text)
    sentences = [s.strip() for s in text.split('<SPLIT>') if s.strip()]
    return sentences


# ======== 从 CSV 构造 corpus 数据 ========

def corpus_data(id2docid, corpus_path):
    """
    从 dongqu.csv 构造样本：
    - 假设有表头: id,location,content
    - 对每一行:
        docid = id2docid[id]
        context: 可以用 location+content，也可以只用 content
        再对 content 切句子，扩展多条
    """
    data = []

    # 注意编码：之前你的文件不是 utf-8，这里用 gb18030 + ignore
    with open(corpus_path, encoding='gb18030', errors='ignore', newline='') as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            raw_id = str(row.get("id", "")).strip()
            location = str(row.get("location", "")).strip()
            content = str(row.get("content", "")).replace('\\n', '\n').replace('\n\n', '\n')

            if not raw_id:
                continue
            if raw_id not in id2docid:
                # 如果映射表里没有这个 id，可以选择跳过或直接用原 id
                # 这里选择直接跳过，以保证 id 一致
                continue

            docid = id2docid[raw_id]

            # 1）整体一条：location + content
            full_text = f"{location} {content}".strip()
            if full_text:
                data.append({"id": docid, "context": full_text})

            # 2）content 按句子切分，每句做多条（保留你原来的 *5 逻辑）
            chunks = split_chinese_sentences(content)
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                # 这里 *5，是模仿你原来 corpus_data 的做法
                for _ in range(5):
                    data.append({"id": docid, "context": chunk})

    return data


# ======== QG 数据（dongqu_queries.jsonl） ========

def qg_data(id2docid, qg_path):
    """
    读取 /QG/dongqu_queries.jsonl
    每行: {"id": "菜品id", "context": "生成的query"}
    映射成: {"id": docid, "context": query}
    """
    data = []
    with open(qg_path, encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_id = str(obj.get("id", "")).strip()
            query = str(obj.get("context", "")).strip()
            if not raw_id or not query:
                continue
            if raw_id not in id2docid:
                continue
            docid = id2docid[raw_id]
            data.append({"id": docid, "context": query})
    return data


# ======== 主函数：构造 train.json / test.json ========

def build_train_test():
    id2docid = load_docid(docid_path)

    print(f"Loaded {len(id2docid)} docid mappings from {docid_path}")

    all_data = []

    # 1) corpus 数据
    corpus = corpus_data(id2docid, corpus_path)
    print(f"Corpus samples: {len(corpus)}")
    all_data += corpus

    # 2) QG 生成的 query 数据
    qg = qg_data(id2docid, qg_path)
    print(f"QG samples: {len(qg)}")
    all_data += qg

    # 打乱
    random.shuffle(all_data)
    total = len(all_data)
    print(f"Total samples: {total}")

    # 简单按 9:1 切分训练 / 测试（你可以改比例）
    split_idx = int(total * 0.9)
    train_datas = all_data[:split_idx]
    test_datas = all_data[split_idx:]

    print(f"Train: {len(train_datas)}, Test: {len(test_datas)}")

    # 写出 train.json
    train_path = os.path.join(output_dir, "train.json")
    with open(train_path, 'w', encoding='utf-8') as fw:
        for line in train_datas:
            fw.write(json.dumps(line, ensure_ascii=False) + '\n')

    # 写出 test.json
    test_path = os.path.join(output_dir, "test.json")
    with open(test_path, 'w', encoding='utf-8') as fw:
        for line in test_datas:
            fw.write(json.dumps(line, ensure_ascii=False) + '\n')

    print("Done!")
    print("Train file:", train_path)
    print("Test file :", test_path)


if __name__ == "__main__":
    build_train_test()
