import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ====== 路径配置：按你的实际情况改 ======
MODEL_DIR = "/home/aiphys/suzitao/IR/output/gr_mt5_base"
DOCID_FILE = "/home/aiphys/suzitao/IR/data/id_title.txt"
CSV_FILE = "/home/aiphys/suzitao/IR/data/QG/total_data.csv"

# dongqu.csv 里的列名：如果不一样，请改成你自己的
CSV_OLD_ID_COL = "id"     # 存 old id 的那一列，比如 1,2,3,...
CSV_LOCATION_COL = "location" # 存位置信息的列名
CSV_CONTENT_COL = "content"   # 存文本内容/描述的列名


# ====== 1. 读取合法 docid 列表，并建立 docid -> old_id 映射 ======
def load_docid_mapping(docid_file):
    """
    docid_file: 形如 id_title.txt
    每行格式: idx<TAB>docid_string
        idx  -> old_id
        docid_string -> 训练/预测用的 docid（长串数字）

    返回:
        docid_to_oldid: {docid_string: old_id(str)}
        all_docids: [docid_string, ...]
        docid_set: set(all_docids)
    """
    docid_to_oldid = {}
    all_docids = []

    with open(docid_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            old_id, docid = parts[0], parts[1]
            docid_to_oldid[docid] = old_id
            all_docids.append(docid)

    docid_set = set(all_docids)
    print(f"[INFO] 共加载合法 docid 数量: {len(docid_set)}")
    return docid_to_oldid, all_docids, docid_set


# ====== 2. 一个简单的“相似度”，用逗号分隔后的 token 的 Jaccard ======
def docid_similarity(a, b):
    sa = set(a.split(","))
    sb = set(b.split(","))
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


# ====== 3. 加载 dongqu.csv，建立 old_id -> (location, content) 映射 ======
def load_dongqu(csv_file):
    """
    假设 csv 有列: old_id, location, content
    如果有多行 old_id 一样，就全部保存成列表
    """

    # 1. 尝试不同编码读取
    encodings_to_try = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    last_err = None
    df = None

    for enc in encodings_to_try:
        try:
            print(f"[INFO] 尝试用编码 {enc} 读取 {csv_file} ...")
            df = pd.read_csv(csv_file, encoding=enc)
            print(f"[INFO] 使用编码 {enc} 读取成功！")
            break
        except UnicodeDecodeError as e:
            print(f"[WARN] 用编码 {enc} 读取失败: {e}")
            last_err = e

    if df is None:
        raise RuntimeError(f"无法用常见编码读取 {csv_file}，最后错误: {last_err}")

    # 2. 打印一下列名，方便你确认
    print("[INFO] dongqu.csv 列名：", list(df.columns))

    if CSV_OLD_ID_COL not in df.columns:
        raise ValueError(f"CSV 中找不到列 '{CSV_OLD_ID_COL}'，请检查列名并在脚本中修改 CSV_OLD_ID_COL。")

    if CSV_LOCATION_COL not in df.columns:
        raise ValueError(f"CSV 中找不到列 '{CSV_LOCATION_COL}'，请检查列名并在脚本中修改 CSV_LOCATION_COL。")

    if CSV_CONTENT_COL not in df.columns:
        raise ValueError(f"CSV 中找不到列 '{CSV_CONTENT_COL}'，请检查列名并在脚本中修改 CSV_CONTENT_COL。")

    oldid_to_rows = {}
    for _, row in df.iterrows():
        old_id = str(row[CSV_OLD_ID_COL])
        location = str(row[CSV_LOCATION_COL])
        content = str(row[CSV_CONTENT_COL])
        oldid_to_rows.setdefault(old_id, []).append((location, content))

    print(f"[INFO] dongqu.csv 中共有 {len(df)} 行数据，old_id 去重后 {len(oldid_to_rows)} 个")
    return oldid_to_rows


# ====== 4. 构建模型 & tokenizer ======
def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[INFO] 使用设备: {device}")
    return tokenizer, model, device


# ====== 5. 针对一个 query 做 beam search -> docid 列表（约束在 docid_set 中） ======
def generate_docids_for_query(
    query,
    tokenizer,
    model,
    device,
    docid_set,
    all_docids,
    topk=10,
    max_input_length=128,
    max_target_length=64,
    raw_beams=50,
):
    """
    返回: [docid1, docid2, ..., docidN] (长度 <= topk，全部在 docid_set 中)
    """

    # 编码输入
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    num_beams = raw_beams
    num_return_sequences = raw_beams

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_target_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )

    # decode 并去重
    candidates = []
    seen = set()
    for i in range(generated.size(0)):
        pred = tokenizer.decode(
            generated[i],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if pred not in seen:
            candidates.append(pred)
            seen.add(pred)

    # 过滤出在 docid_set 中的
    valid_preds = []
    for cand in candidates:
        if cand in docid_set and cand not in valid_preds:
            valid_preds.append(cand)
        if len(valid_preds) >= topk:
            break

    # 如果还不够 topk 个，可以用相似度从 all_docids 里补足
    if len(valid_preds) < topk:
        if len(candidates) > 0:
            base = candidates[0]
        else:
            base = ""  # 极端情况

        scored = []
        for docid in all_docids:
            if docid in valid_preds:
                continue
            sim = docid_similarity(base, docid)
            scored.append((docid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        for docid, _ in scored:
            valid_preds.append(docid)
            if len(valid_preds) >= topk:
                break

    return valid_preds[:topk]


# ====== 6. 交互式对话检索 ======
def interactive_loop(
    tokenizer,
    model,
    device,
    docid_to_oldid,
    docid_set,
    all_docids,
    oldid_to_rows,
    topk=10,
):
    print("\n===== 交互式食堂检索开始 =====")
    print("输入你的查询（例如：\"想吃辣的面\"、\"东区二楼有什么饮品\"）。")
    print("输入 q / quit / exit 退出。\n")

    while True:
        query = input("你: ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            print("再见~")
            break

        if not query:
            continue

        # 1) 用 GR 模型生成 docid 列表
        docids = generate_docids_for_query(
            query=query,
            tokenizer=tokenizer,
            model=model,
            device=device,
            docid_set=docid_set,
            all_docids=all_docids,
            topk=topk,
        )

        if not docids:
            print("系统: 暂时没有找到合适的结果。试试换个说法？\n")
            continue

        # 2) 把 docid 映射为 old_id，再查 dongqu.csv 的 location & content
        print("\n系统: 给你找到这些结果（top-{}）：\n".format(topk))
        rank = 1
        for docid in docids:
            old_id = docid_to_oldid.get(docid)
            if old_id is None:
                continue

            rows = oldid_to_rows.get(old_id, [])
            if not rows:
                continue

            for (location, content) in rows:
                print(f"#{rank}")
                print(f"[位置] {location}")
                print(f"[内容] {content}")
                print(f"[old_id] {old_id}")
                print(f"[docid] {docid}")
                print("-" * 40)
                rank += 1

                if rank > topk:
                    break
            if rank > topk:
                break

        if rank == 1:
            print("系统: 找到了 docid，但在 dongqu.csv 中没对应记录，检查一下 old_id / CSV 映射？")

        print("\n")  # 一点空行分隔下一轮


def main():
    # 加载映射 & 数据
    docid_to_oldid, all_docids, docid_set = load_docid_mapping(DOCID_FILE)
    oldid_to_rows = load_dongqu(CSV_FILE)

    # 加载模型
    tokenizer, model, device = load_model_and_tokenizer(MODEL_DIR)

    # 进入交互式循环
    interactive_loop(
        tokenizer=tokenizer,
        model=model,
        device=device,
        docid_to_oldid=docid_to_oldid,
        docid_set=docid_set,
        all_docids=all_docids,
        oldid_to_rows=oldid_to_rows,
        topk=10,
    )


if __name__ == "__main__":
    main()
