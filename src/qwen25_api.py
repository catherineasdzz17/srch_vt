import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from llm_helpers import QWEN25


LLM_MODEL = QWEN25()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# 生成结构化元素实体信息
@app.post("/generate_elements")
async def generate_elements(request: dict):
    text = request.get('text', '')
    result = LLM_MODEL.generate_elements_str(text)
    return {"result": result}

# 判断两个句子哪个更通顺
@app.post('/judge_sim_tra')
async def judge_sim_tra(request: dict):
    sim = request.get('sim', '')
    tra = request.get('tra', '')
    ans = LLM_MODEL.judge_sim_tra(sim, tra)
    return {"result": ans}

# 分段及添加标点
@app.post("/add_punctuation")
async def add_punctuation(request: dict):
    text = request.get('text', '')
    result = LLM_MODEL.add_punctuation(text)
    return {"result": result}

# 判断两个句子是否匹配
@app.post("/judge_similarity")
async def judge_similarity(request: dict):
    a = request.get('a', '')
    b = request.get('b', '')
    result = LLM_MODEL.judge_similarity(a, b)
    return {"result": result}

# 生成结构化隐喻讽刺实体信息
@app.post("/generate_yinyu_fengci")
async def generate_yinyu_fengci(request: dict):
    text = request.get('text', '')
    result = LLM_MODEL.generate_yinyu_fengci(text)
    return {"result": result}

# 生成摘要
@app.post("/generate_summary")
async def generate_summary(request: dict):
    text = request.get('text', '')
    element = request.get('element', [])
    result = LLM_MODEL.generate_summary(text, element)
    return {"summary": result}

# 生成分段摘要
@app.post("/generate_summary_chunk")
async def generate_summary_chunk(request: dict):
    text = request.get('text', '')
    element = request.get('element', [])
    result = LLM_MODEL.generate_summary_chunk(text, element)
    return {"summary": result}


if __name__ == "__main__":
    # 配置Uvicorn运行FastAPI应用
    with torch.no_grad():
        logging.disable(logging.DEBUG)
        config = uvicorn.Config(app, host="0.0.0.0", port=14700, log_level="info")
        server = uvicorn.Server(config)
        server.run()