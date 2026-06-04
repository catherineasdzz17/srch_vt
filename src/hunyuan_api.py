import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from translation_helpers import HunyuanMT


TRANSLATOR = HunyuanMT()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/trans2zh") 
async def trans2zh(request: dict):
    text = request.get('text', '')
    text = str(text or "")
    result = TRANSLATOR.trans2zh(text)
    return {"translation": str(result)}


if __name__ == "__main__":
    with torch.no_grad():
        logging.disable(logging.DEBUG)
        config = uvicorn.Config(app, host="0.0.0.0", port=14800, log_level="info")
        server = uvicorn.Server(config)
        server.run()