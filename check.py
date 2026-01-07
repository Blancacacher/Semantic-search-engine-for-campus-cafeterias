import argparse


def parse_results_file(path):
    """
    解析 test_pred_top50_constrained.txt
    返回一个列表，每个元素是 dict:
    {
        "truth": truth_docid_str,
        "preds": [pred_1, pred_2, ..., pred_k]
    }
    """
    results = []

    with open(path, "r", encoding="utf-8") as f:
        cur_truth = None
        cur_preds = []

        for line in f:
            line = line.strip()
            if not line:
                # 一个样本结束
                if cur_truth is not None:
                    results.append({"truth": cur_truth, "preds": cur_preds})
                    cur_truth = None
                    cur_preds = []
                continue

            if line.startswith("truth:"):
                cur_truth = line[len("truth:"):].strip()
            elif line.startswith("predict_"):
                # 行格式: predict_i: docid
                parts = line.split(":", 1)
                if len(parts) == 2:
                    pred = parts[1].strip()
                    cur_preds.append(pred)

        # 文件末尾如果没有空行，补上最后一个样本
        if cur_truth is not None:
            results.append({"truth": cur_truth, "preds": cur_preds})

    return results


def compute_mrr_recall(results, ks=(1, 5, 10, 20, 50)):
    """
    results: parse_results_file 的输出
    ks: 需要计算的 k 列表
    返回两个 dict: mrr_at, recall_at
    """
    ks = sorted(ks)
    mrr_sum = {k: 0.0 for k in ks}
    recall_cnt = {k: 0 for k in ks}

    num_queries = len(results)

    for item in results:
        truth = item["truth"]
        preds = item["preds"]

        # 找到 truth 在 preds 中的 rank（1-based），没命中则为 None
        rank = None
        for i, p in enumerate(preds):
            if p == truth:
                rank = i + 1   # rank 从 1 开始
                break

        for k in ks:
            if rank is not None and rank <= k:
                recall_cnt[k] += 1
                mrr_sum[k] += 1.0 / rank

    mrr_at = {k: (mrr_sum[k] / num_queries if num_queries > 0 else 0.0) for k in ks}
    recall_at = {k: (recall_cnt[k] / num_queries if num_queries > 0 else 0.0) for k in ks}

    return mrr_at, recall_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/home/aiphys/suzitao/IR/data/train_title/test_pred_top100_constrained.txt",
        help="预测结果文件路径",
    )
    args = parser.parse_args()

    results = parse_results_file(args.input)
    print(f"共解析到 {len(results)} 条样本")

    ks = [1, 5, 10, 20, 50, 100]
    mrr_at, recall_at = compute_mrr_recall(results, ks=ks)

    print("===== MRR@k =====")
    for k in ks:
        print(f"MRR@{k}: {mrr_at[k]:.6f}")

    print("\n===== Recall@k =====")
    for k in ks:
        print(f"Recall@{k}: {recall_at[k]:.6f}")


if __name__ == "__main__":
    main()
