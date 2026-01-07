import os
import time
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================= 配置区域 =================
# 请根据你的实际路径修改这里
class Config:
    # 模型路径 (checkpoint-2370 或者 gr_mt5_base)
    MODEL_DIR = "./model" 
    # 映射文件路径
    DOCID_FILE = "./data/id_title.txt"
    # 数据内容文件路径
    CSV_FILE = "./data/total_data.csv"
    
    # CSV 列名配置 (与 interactive_gr_retrieval.py 保持一致)
    CSV_OLD_ID_COL = "id"
    CSV_LOCATION_COL = "location"
    CSV_CONTENT_COL = "content"

    # 生成参数
    TOP_K = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="WeChat MiniProgram Search API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# ================= 辅助函数 (源自你的脚本) =================

def load_docid_mapping(docid_file):
    docid_to_oldid = {}
    all_docids = []
    with open(docid_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) != 2: continue
            old_id, docid = parts[0], parts[1]
            docid_to_oldid[docid] = old_id
            all_docids.append(docid)
    return docid_to_oldid, all_docids, set(all_docids)

def load_dongqu(csv_file):
    # 尝试多种编码读取
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_file, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise RuntimeError(f"无法读取 CSV 文件: {csv_file}")

    # 建立 old_id -> list of details 映射
    oldid_to_rows = {}
    for _, row in df.iterrows():
        # 确保转为字符串处理
        old_id = str(row[Config.CSV_OLD_ID_COL])
        location = str(row[Config.CSV_LOCATION_COL])
        content = str(row[Config.CSV_CONTENT_COL])
        
        if old_id not in oldid_to_rows:
            oldid_to_rows[old_id] = []
        oldid_to_rows[old_id].append({"location": location, "content": content})
            
    return oldid_to_rows

def docid_similarity(a, b):
    sa = set(a.split(","))
    sb = set(b.split(","))
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0

# ================= 核心服务类 =================

class SearchService:
    def __init__(self):
        print(f"[Init] 正在加载模型和数据，使用设备: {Config.DEVICE} ...")
        
        # 1. 加载数据映射
        self.docid_map, self.all_docids, self.docid_set = load_docid_mapping(Config.DOCID_FILE)
        self.content_map = load_dongqu(Config.CSV_FILE)
        
        # 2. 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_DIR)
        self.model.to(Config.DEVICE)
        self.model.eval()
        
        print("[Init] 服务加载完成!")

    def search(self, query: str, topk: int = 10):
        # 1. 编码
        inputs = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=128
        ).to(Config.DEVICE)
        
        # 2. 生成 (Beam Search)
        # 稍微减少 beams 数量以提高 API 响应速度，例如设为 20
        num_beams = 20 
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True
            )
            
        # 3. 解码并过滤
        candidates = []
        seen = set()
        for i in range(generated.size(0)):
            pred = self.tokenizer.decode(generated[i], skip_special_tokens=True).strip()
            if pred not in seen:
                candidates.append(pred)
                seen.add(pred)

        # 4. 验证有效性
        valid_docids = []
        for cand in candidates:
            if cand in self.docid_set:
                valid_docids.append(cand)
            if len(valid_docids) >= topk:
                break
        
        # 5. 补全 (如果不足 topk，使用相似度 - 可选，为了速度可注释掉这部分)
        if len(valid_docids) < topk and candidates:
            base = candidates[0]
            scored = []
            # 注意：全量计算相似度可能较慢，生产环境建议优化
            for docid in self.all_docids:
                if docid in valid_docids: continue
                sim = docid_similarity(base, docid)
                scored.append((docid, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            for docid, _ in scored:
                valid_docids.append(docid)
                if len(valid_docids) >= topk: break

        # 6. 组装最终结果
        results = []
        for docid in valid_docids:
            old_id = self.docid_map.get(docid)
            if old_id and old_id in self.content_map:
                items = self.content_map[old_id]
                for item in items:
                    results.append({
                        "docid": docid,
                        "old_id": old_id,
                        "location": item["location"],
                        "content": item["content"]
                    })
                    if len(results) >= topk: break
            if len(results) >= topk: break
            
        return results

# ================= 全局实例 =================
# 在应用启动时初始化，常驻内存
search_service = SearchService()

# ================= API 定义 =================

class QueryRequest(BaseModel):
    query: str

class SearchResult(BaseModel):
    location: str
    content: str
    docid: str
    old_id: str

class ResponseModel(BaseModel):
    code: int
    message: str
    data: List[SearchResult]

@app.post("/search", response_model=ResponseModel)
async def search_endpoint(request: QueryRequest):
    if not request.query.strip():
        return {"code": 400, "message": "Query cannot be empty", "data": []}
    
    try:
        # 调用服务进行搜索
        results = search_service.search(request.query, topk=Config.TOP_K)
        return {
            "code": 200,
            "message": "success",
            "data": results
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"code": 500, "message": str(e), "data": []}

# 启动命令 (在终端运行):
# uvicorn main:app --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)