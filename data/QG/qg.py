import os
import csv
import json
from tqdm import tqdm
from openai import OpenAI

# ========= 配置区域 =========

# 从环境变量里读 DeepSeek 的 API Key
DEEPSEEK_API_KEY = "sk-87525572add64abf88778f9c137a23cd"
if not DEEPSEEK_API_KEY:
    raise ValueError("请先在环境变量中设置 DEEPSEEK_API_KEY")

# DeepSeek API 兼容 OpenAI，用这个 base_url
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

MODEL_NAME = "deepseek-chat"   # 官方推荐聊天模型

# system prompt：定义模型角色
SYSTEM_PROMPT = """你是一个为高校食堂检索系统构造训练数据的助手。
用户会在搜索框里输入查询语句（query），想要找到他们想吃的菜品。
你会看到一条关于某个菜品的描述，需要根据这条描述，生成若干个“学生可能会如何搜索”的自然语言查询。
注意：这些查询是“搜索词”，不是商品标题，也不是长篇文案。"""

# ========= 生成 query 的函数 =========

def generate_queries_for_content(content: str, num_queries: int = 10):
    """
    给定一条菜品描述 content，调用 DeepSeek 生成 num_queries 条查询。
    返回 list[str]
    """
    user_prompt = f"""
下面是一条关于高校食堂菜品的描述：

---
{content}
---

请你从“学生在食堂检索系统中会怎么搜索”的角度，生成 {num_queries} 条中文查询（query）。
要求：
1. 每条查询是一句简短的自然语言或关键词组合，长度一般不超过 40 个汉字。
2. 查询要多样化，尽量覆盖不同的表达方式，例如：
   - 直接搜菜名（如果有）
   - 按口味 / 做法 / 主料
   - 按价格 / 份量 / 是否实惠
   - 按位置（哪栋楼、哪层、哪个窗口）
3. 不要包含序号、括号或其它多余符号，不要加引号。
4. 不要复述“这是一道什么菜”的说明，只输出用户可能在搜索框输入的话。

输出格式必须是一个 JSON 对象，形如：
{{
  "queries": [
    "第一条查询",
    "第二条查询",
    ...
  ]
}}

只输出这个 JSON 对象，不要输出任何解释性文字。
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # 要求以 JSON 形式返回，方便解析
        response_format={"type": "json_object"},
        temperature=0.7,
        top_p=0.9,
    )

    content_str = response.choices[0].message.content
    data = json.loads(content_str)

    queries = data.get("queries", [])
    # 保证是字符串列表
    queries = [str(q).strip() for q in queries if str(q).strip()]
    # 截断到 num_queries 条
    return queries[:num_queries]

# ========= 主程序：读 CSV，写 JSONL =========

def main():
    csv_path = "./total_data.csv"              # 你的原始数据
    output_path = "./total_queries.jsonl" # 输出 JSONL

    # 视你的文件编码而定：如果之前是 gb18030，就用这个
    with open(csv_path, "r", encoding="gb18030", errors="ignore", newline="") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)  # 要求表头包含 id, location, content

        for row in tqdm(reader, desc="Generating queries"):
            dish_id = str(row["id"])
            content = str(row["content"])

            # 调用 DeepSeek 生成 10 条 query
            try:
                queries = generate_queries_for_content(content, num_queries=15)
            except Exception as e:
                print(f"[ERROR] id={dish_id} 调用 API 失败：{e}")
                continue

            # 写成 JSONL：每条 query 一行
            # 这里按你的要求：同一个 id 生成 10 行，id 相同
            for q in queries:
                record = {
                    "id": dish_id,
                    "context": q  # 你说字段叫 context，这里就用 context 存 query
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("完成！输出文件：", output_path)


if __name__ == "__main__":
    main()
