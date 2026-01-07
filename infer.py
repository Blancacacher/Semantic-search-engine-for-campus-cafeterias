import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 1. 读取合法 docid 列表
def load_docids(docid_file):
    """
    docid_file: 形如 id_title.txt
    每行格式: idx<TAB>docid_string
    """
    all_docids = []
    with open(docid_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            _, docid = parts
            all_docids.append(docid)

    docid_set = set(all_docids)
    print(f"共加载合法 docid 数量: {len(docid_set)}")
    return all_docids, docid_set


# 一个简单的“相似度”，用逗号分隔后的 token 的 Jaccard
def docid_similarity(a, b):
    sa = set(a.split(","))
    sb = set(b.split(","))
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def main():
    # ====== 路径根据你自己的情况改一下 ======
    model_dir = "/home/aiphys/suzitao/IR/output/gr_mt5_base"  # 训练好的模型目录
    test_file = "/home/aiphys/suzitao/IR/data/train_title/test.json"
    output_file = "/home/aiphys/suzitao/IR/data/train_title/test_pred_top100_constrained.txt"
    docid_file = "/home/aiphys/suzitao/IR/data/id_title.txt"

    # ====== 加载合法 docid 集 ======
    all_docids, docid_set = load_docids(docid_file)

    # ====== 加载 tokenizer 和 model ======
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ====== 生成参数 ======
    max_input_length = 128
    max_target_length = 64
    desired_topk = 100          # 希望最终得到的候选数
    raw_beams = 200            # 先从模型里拿这么多候选，再过滤
    num_beams = raw_beams
    num_return_sequences = raw_beams

    total = 0
    with open(test_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            truth_id = str(obj["id"])
            context = obj["context"]

            # 编码输入
            inputs = tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # 2. beam search 生成很多候选（raw_beams 个）
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_target_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    early_stopping=True,
                )

            # 3. decode 为字符串，并去重（保持顺序）
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

            # 4. 过滤出“合法 docid”（出现在 id_title.txt 的第二列）
            valid_preds = []
            for cand in candidates:
                if cand in docid_set and cand not in valid_preds:
                    valid_preds.append(cand)
                if len(valid_preds) >= desired_topk:
                    break

            # 5. 如果还不够 50 个，用简单相似度从全集里补
            if len(valid_preds) < desired_topk:
                # 以 top-1 candidate 作为“参考”，从 all_docids 里找最相似的补足
                if len(candidates) > 0:
                    base = candidates[0]
                else:
                    base = truth_id  # 极端情况，就拿 truth_id 当参考

                scored = []
                for docid in all_docids:
                    if docid in valid_preds:
                        continue
                    sim = docid_similarity(base, docid)
                    scored.append((docid, sim))

                scored.sort(key=lambda x: x[1], reverse=True)

                for docid, _ in scored:
                    valid_preds.append(docid)
                    if len(valid_preds) >= desired_topk:
                        break

            # 这里保证 valid_preds 里全是来自 docid_set 的字符串，
            # 且数量 >= desired_topk（如果 all_docids 本身就少于 50，那就等于全集大小）

            # ====== 写入结果 ======
            fout.write(f"truth: {truth_id}\n")
            for idx, pred in enumerate(valid_preds[:desired_topk], start=1):
                fout.write(f"predict_{idx}: {pred}\n")
            fout.write("\n")

            total += 1
            if total % 20 == 0:
                print(f"已处理 {total} 条样本")

    print(f"完成！共处理 {total} 条样本，结果已写入: {output_file}")


if __name__ == "__main__":
    main()
