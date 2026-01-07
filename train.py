import os
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset, random_split

from transformers import (
    MT5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


# ======================
# 自定义数据集
# ======================

class GRDataset(Dataset):
    """
    读取 jsonlines 文件，每行形如：
    {"id": "...", "context": "..."}
    将 context 作为输入，id 作为输出标签。
    """
    def __init__(
        self,
        file_path: str,
        tokenizer: T5TokenizerFast,
        max_input_length: int = 128,
        max_target_length: int = 64,
    ):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # 这里假设字段名固定为 id 和 context
                _id = str(obj["id"])
                context = obj["context"]
                self.data.append({"id": _id, "context": context})

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        context = example["context"]
        target_id = example["id"]

        # 编码输入：context
        model_inputs = self.tokenizer(
            context,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 编码输出：id（用字符串形式）
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_id,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)

        labels_ids = labels["input_ids"].squeeze(0)
        # Trainer 约定：把 pad_token_id 替换成 -100，这样不会计算 loss
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
        }


# ======================
# 主训练逻辑
# ======================

def main():
    # -------- 路径配置（按你的环境）--------
    model_name_or_path = "/home/aiphys/suzitao/models/mt5-base"
    train_file = "/home/aiphys/suzitao/IR/data/train_title/train.json"
    output_dir = "/home/aiphys/suzitao/IR/output/gr_mt5_base"

    os.makedirs(output_dir, exist_ok=True)

    # -------- 加载 tokenizer & model --------
    tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path)

    # 如果想把逗号分隔的 id 当作一个整体 token，可以在这里添加特殊 token
    # 目前我们直接用原始 id 字符串，会被拆成子词/数字 token，也可以正常训练。
    # special_tokens = {"additional_special_tokens": ["<DOC>"]}
    # tokenizer.add_special_tokens(special_tokens)
    # model.resize_token_embeddings(len(tokenizer))

    # -------- 构造数据集 --------
    dataset = GRDataset(
        file_path=train_file,
        tokenizer=tokenizer,
        max_input_length=128,
        max_target_length=64,
    )

    # 简单划分一小部分做验证集（比如 98% 训练，2% 验证）
    train_size = int(0.98 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # -------- 训练参数 --------
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=3,

        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=1000,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )


    # -------- 定义 Trainer --------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # -------- 开始训练 --------
    trainer.train()

    # -------- 保存最终模型 --------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("训练完成，模型已保存到:", output_dir)


if __name__ == "__main__":
    main()
