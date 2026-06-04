import os
import ctypes

clean_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = clean_ld_path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import logging
import modelscope
import face_recognition

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib64:" + clean_ld_path

try:
    ctypes.CDLL("/usr/local/cuda-12.4/lib64/libcudart.so.12")
    ctypes.CDLL("/usr/local/cuda-12.4/lib64/libcublas.so.12")
except Exception as e:
    pass

print("All Ready!")

import asyncio
import os.path
import paddle
import time
import torch
import uvicorn
import uuid
import numpy as np

from modelscope import Tasks
from pydantic import BaseModel
from paddleocr import PaddleOCR
from transformers import ViTImageProcessor, AutoModelForSpeechSeq2Seq, AutoProcessor
from typing import Optional, List, Dict, Any
from collections import deque
from dataclasses import dataclass
from FlagEmbedding import BGEM3FlagModel
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.param_functions import Form
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from operations.count import do_count
from operations.drop import do_drop
from operations.graph_search import SearchGraph
from operations.keywords_search import TFIDFSearch
from operations.load import do_load, do_img_load
from operations.search import search_in_milvus, do_search, mulit_search

from audio_handler import transcribe_long_audio
from bge_m3_helper import search_text
from encode import ChineseCLIP
from img_helpers import img_recg
from logs import LOGGER
# from milvus_helpers import MilvusHelper
from my_utils import element_filter, call_qwen, trans_ocr
# from mysql_helpers import MySQLHelper
from segment_helpers import Segment_generator, get_segment_images, get_imgs
from video_handler import video_handler_main, chinese_ner
from vit_model import vit_model
from config import UPLOAD_IMAGE_PATH, SEGMENT_SAVE_PATH, TFIDF_PATH, CLASS_CHECKPOINT, CLASS_MODEL_PATH, \
    UPLOAD_VIDEO_PATH, AUDIO_MODEL_PATH, UPLOAD_AUDIO_PATH
from fl_handler import fl_video_main, fl_long_main, cleanup_gpu_memory_enhanced


logging.disable(logging.DEBUG)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# 存储任务状态
tasks: Dict[str, dict] = {}


if not os.path.exists(UPLOAD_IMAGE_PATH):
    os.makedirs(UPLOAD_IMAGE_PATH)
    LOGGER.info(f"mkdir the path:{UPLOAD_IMAGE_PATH}")


class Item(BaseModel):
    Table: Optional[str] = None
    File: str


class LoginFormVo(BaseModel):
    account: str
    password: str


class LoginResultDto(BaseModel):
    id: int = 1
    account: str = ''
    token: str = ''


class SearchData(BaseModel):
    inputText: str = ''
    selectSearchType: str = ''
    imageFiles: List = []
    type: List = []
    conditions: Dict = {}


@app.get('/')
async def root():
    '''
    根目录访问接口
    用于确认FastAPI服务是否成功启动
    '''
    return {"message": "Welcome!"}


@app.post('/user/login')
async def login(loginFormVo: LoginFormVo):
    '''
    用户登录接口
    传入一个LoginFormVo对象，返回一个LoginResultDto对象
    '''
    print(loginFormVo)
    loginResultDto = LoginResultDto()
    loginResultDto.id = 1
    loginResultDto.account = loginFormVo.account
    loginResultDto.token = 'dfefedwf12ff'
    print(loginResultDto)
    return {'data': loginResultDto}


@app.get('/user/permission')
async def getPermission():
    '''
    获取权限列表
    用于前端用户权限控制（菜单权限、按钮权限等）
    '''
    permission = ["permission.browse", "permission.create", "permission.edit", "permission.remove"]
    return {'data': permission}


@app.get('/search/text/elements')
async def do_search_text_elements(query_sentence: str = None, top_k: int = 9):
    '''
    文本查询
    '''
    # 调用Qwen抽取元素
    print(query_sentence)
    elements_response = call_qwen(query_sentence)[:500]
    print(elements_response)

    # 搜索相关文本，并通过文本找图（如有）
    texts = []
    a = time.time()
    # 使用BGE-M3模型对query_sentence编码，并与text_set向量集合做相似度检索
    text_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=text_set)
    texts = [eve[1] for eve in text_ans]  # 存储原始文本内容
    # 与img_text_set做相似度检索，存储图像路径和对应文本
    img_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=img_text_set)
    images = [{'image': os.path.join(os.getcwd(), 'candidate_set/images',eve[1]), 'text': eve[2]} for eve in img_ans]
    # 与video_text_set做相似度检索，存储视频路径和对应文本
    videos_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=video_text_set)
    videos = [{'video': os.path.join(os.getcwd(), 'candidate_set/videos',eve[1]), 'text': eve[2]} for eve in videos_ans]
    # 与audio_text_set做相似度检索，存储音频路径和对应文本
    audios_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=audio_text_set)
    audios = [{'audio': os.path.join(os.getcwd(), 'candidate_set/audios',eve[1]), 'text': eve[2]} for eve in audios_ans]
    
    # 对Qwen生成的元素结果做中文命名实体识别，并进行实体类别映射
    names = chinese_ner(elements_response, ner_pipeline=ner_pipeline)
    mp = {'PER': ["Politicians"], 'LOC': ["Nation"], 'ORG': ["Orgnization"]}

    graph_neighbors = {}
    for key in names.keys():  # 遍历所有实体类型
        skind = mp[key]  # 类别
        for name in names[key]:  # 遍历所有实体名
            flag = False
            for element_eve in elements_set:  # 遍历图谱中的所有实体块
                for element_eve2 in element_eve:  # 每个块中可能包含多个实体信息（嵌套结构）
                    # 找到与当前实体名和类型都一致的图谱节点
                    if name in element_eve2['name'] and skind==element_eve2['labels']:
                        graph_neighbors[name] = element_eve  # 将整个实体块加入
                        flag = True
                        break
                if flag:
                    break

    LOGGER.info("Successfully searched similar text!")
    # 以下为自用前端返回结果，可选
    # res_images形式：[{'img':'','text':''}]
    # final_texts形式：[{'text':'','imgs':['']}]
    # texts = [{'text':eve,'imgs':[]} for eve in texts]
    # images = [{'img':eve['image'],'text':eve['text']} for eve in images]
    # print(videos)
    # print(audios)

    print("elements_response:", elements_response)
    print("texts:", texts)

    # 调用Qwen生成与查询语句相关的讽喻关键词
    yinyufengci = call_qwen(query_sentence, ctype='generate_yinyu_fengci')
    texts.insert(0, elements_response)
    yinyufengci = yinyufengci.replace(";;","--------")
    texts = [yinyufengci]+texts
    print(yinyufengci)
    return {
        'elements': elements_response,  # Qwen生成的结构元素
        'texts': texts,  # 匹配到的文本列表
        'images': images,  # 匹配图像
        'videos': videos,  # 匹配视频
        'audios': audios,  # 匹配音频
        'neibors': graph_neighbors  # 命名实体对应图谱邻接节点
    }


@app.post('/search/image/elements')
async def do_search_image_elments(file: List[UploadFile] = File(...), top_k: int = Form(9), condition: str = Form('')):
    '''
    图像查询
    '''
    try:
        lists = [i.filename for i in file]
        print(lists)
        # 初始化后端需要保存上传图片/视频文件的位置
        dirs = UPLOAD_IMAGE_PATH
        segment_path = SEGMENT_SAVE_PATH
        # 原图，保存在tmp内
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        # 切割后的图，保存在tmp内
        if not os.path.exists(segment_path):
            os.makedirs(segment_path)

        segment_path = os.path.join(os.getcwd(), segment_path)
        imgs_path = []
        # 写入上传的图片
        for i in file:
            img_path = os.path.join(os.getcwd(), os.path.join(dirs, i.filename))
            imgs_path.append(img_path)
            with open(img_path, 'wb+') as f:
                f.write(await i.read())

        # 图像特征向量提取
        elements = []
        key_words = []
        image_feature = CLIP_MODEL.clip_extract_features(Image.open(imgs_path[0]), type='image')
        imgs = mulit_search(image_feature[0], 5, 'img')  # 检索相似图像
        imgs = [{'image': os.path.join(os.getcwd(), 'candidate_set/images', eve), 'text': ''} for eve in imgs]

        # 使用SAM进行图像分割
        for i in os.listdir(segment_path):
            os.remove(os.path.join(segment_path, i))
        get_segment_images(SAM, imgs_path[0], segment_path)  # 图像分割
        print(os.listdir(segment_path))
        # 分割后的图
        pics, fs = get_imgs(segment_path)
        print(pics)
        # 对分割完的子图进行识别，element_dict为元素列表
        for pic in pics:
            element_dict, key_word = img_recg(processor, class_model, CLIP_MODEL, pic)  # 逐个子图识别元素
            key_words = key_words + key_word
            if element_dict is not None:
                elements.append(element_dict)
        elements = element_filter(elements)  # 去除无效元素，去除国家元素（过于常见信息量较低）
        print(elements)
        
        # 构建用于查询的query sentence
        frame = np.array(Image.open(imgs_path[0]).convert('RGB'))
        # 对图像进行OCR识别
        results_ch = ocr_ch.ocr(frame, cls=True)[0]  # 简体中文识别
        if results_ch is not None:
            results_ch = [result[1][0] for result in results_ch if result[1][1]>0.8]  # 提取置信度>0.8的识别结果
            results_chinese_cht = ocr_chinese_cht.ocr(frame, cls=True)[0]  # 繁体中文识别
            if results_chinese_cht is not None:
                results_chinese_cht = [result[1][0] for result in results_chinese_cht if result[1][1]>0.8]
            else:
                results_chinese_cht = []
            # 将繁体简体的ocr结果配对处理
            result_lst = trans_ocr(results_ch, results_chinese_cht)  
            # 判断一个文本是否为繁简体的语义重复表达，并保留有用信息
            result_lst = [call_qwen(eve, ctype='judge_sim_tra') for eve in result_lst]  
            # 将处理后的结果拼接为查询输入
            query_sentence = ','.join(result_lst)
        else:
            # 未识别出文本则使用elements构建
            query_lst = [dic['elements'][0] for dic in elements]
            query_sentence = ','.join(query_lst)
        # 若识别出的太短，则使用elements补充
        if len(query_sentence.strip())==0 or len(query_sentence)<7:
            query_lst = [dic['elements'][0] for dic in elements]
            query_sentence = ','.join(query_lst)
        # 将元素再次加入查询句做加强
        for element in elements:
            query_sentence = query_sentence + ',' + element['elements'][0]
        print(query_sentence)

        # 使用query_sentence检索各模态相似内容
        text_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=text_set)
        texts = [eve[1] for eve in text_ans]
        img_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=img_text_set)[:5]
        images = imgs + [{'image': os.path.join(os.getcwd(), 'candidate_set/images', eve[1]), 'text': eve[2]} for eve in img_ans]
        videos_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=video_text_set)
        videos = [{'video': os.path.join(os.getcwd(), 'candidate_set/videos', eve[1]), 'text': eve[2]} for eve in videos_ans]
        audios_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=audio_text_set)
        audios = [{'audio': os.path.join(os.getcwd(), 'candidate_set/audios', eve[1]), 'text': eve[2]} for eve in audios_ans]

        # 倒排索引搜索文本语料
        LOGGER.info("Successfully searched similar text!")
        # res_images形式：[{'img':'','text':''}]
        # final_texts形式：[{'text':'','imgs':['']}]
        # texts = [{'imgs':[],'text':eve} for eve in texts]
        # images = [{'img':eve['image'],'text':eve['text']} for eve in images]
        # '
        # print(videos)
        # print(audios)
        # print(images)
        return {
            'elements': elements,
            'texts': texts,
            'images': images, 
            'videos': videos,
            'audios': audios, 
            'query': query_sentence
        }

    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/search/video/elements')
async def do_search_video_elments(file: List[UploadFile] = File(...), top_k: int = Form(9), condition: str = Form('')):
    '''
    视频查询
    '''
    # 加载视频
    lists = [i.filename for i in file]
    print(lists)
    video_dir = UPLOAD_VIDEO_PATH
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    for i in file:
        video_path = os.path.join(video_dir, str(i.filename))
        with open(video_path, 'wb+') as f:
            f.write(await i.read())
            # 视频处理，提取关键帧、识别元素、文字OCR
            element_dicts, query_sentence = video_handler_main(video_path, sam=SAM, processor=processor, class_model=class_model,
                             CLIP_MODEL=CLIP_MODEL, ocr_chinese_cht=ocr_chinese_cht, ocr_ch=ocr_ch,
                             sentence_model=sentence_model, ner_pipeline=ner_pipeline)

    # 使用query_sentence检索各模态相似内容
    text_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=text_set)
    texts = [eve[1] for eve in text_ans]
    img_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=img_text_set)
    images = [{'image': os.path.join(os.getcwd(), 'candidate_set/images', eve[1]), 'text': eve[2]} for eve in img_ans]
    videos_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=video_text_set)
    videos = [{'video': os.path.join(os.getcwd(), 'candidate_set/videos', eve[1]), 'text': eve[2]} for eve in videos_ans]
    audios_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=audio_text_set)
    audios = [{'audio': os.path.join(os.getcwd(), 'candidate_set/audios', eve[1]), 'text': eve[2]} for eve in audios_ans]
    # print(ans)
    # texts = [{'imgs':[],'text':eve} for eve in texts]
    # images = [{'img':eve['image'],'text':eve['text']} for eve in images]
    # print(videos)
    # print(audios)
    print(query_sentence)
    print(images)

    return {
        'elements': element_dicts, 
        'images': images, 
        'texts': texts, 
        'videos': videos, 
        'audios': audios, 
        'query': query_sentence
    }


# @app.post("/video_handler")
# async def fl_video_handler(file: UploadFile = File(...)):
#     '''
#     视频处理 (同步接口，保留向后兼容)
#     '''
#     start_time = time.time()
#     video_dir = "fl_tmp"
#     os.makedirs(video_dir, exist_ok=True)
#     video_path = os.path.join(video_dir, file.filename)

#     ocr_fl = None

#     try:
#         with open(video_path, "wb") as f:
#             f.write(await file.read())

#         print("创建OCR实例...")
#         ocr_fl = PaddleOCR(
#             use_angle_cls=True,
#             lang="ch",
#             show_log=False,
#             use_gpu=True,
#             gpu_mem=500,
#             enable_mkldnn=False,
#             use_tensorrt=False,
#         )
#         print("OCR实例创建成功")

#         result = fl_long_main(
#             input_video=video_path,
#             work_dir="fl_tmp_seg",
#             audio_model=audio_model,
#             audio_processor=audio_processor,
#             sam=SAM,
#             class_model=class_model,
#             processor=processor,
#             CLIP_MODEL=CLIP_MODEL,
#             ocr_fl=ocr_fl,
#             sentence_model=sentence_model,
#             ner_pipeline=ner_pipeline,
#             do_search=True
#         )

#         end_time = time.time()
#         elapsed_seconds = round(end_time - start_time, 2)
#         print("总耗时: ", elapsed_seconds)

#         return result

#     finally:
#         if ocr_fl is not None:
#             try:
#                 print("清理OCR实例...")
#                 del ocr_fl
#                 import gc
#                 gc.collect()
#                 try:  # 清理GPU缓存
#                     import paddle
#                     paddle.device.cuda.empty_cache()
#                 except:
#                     pass
#                 print("OCR实例清理完成")
#             except Exception as e:
#                 print(f"OCR清理失败: {e}")
        
#         cleanup_temp_files(video_path)


# @app.post("/video_handler/submit")
# async def submit_video_task(file: UploadFile = File(...)):
#     '''
#     提交视频处理任务 (异步接口)
#     '''
#     task_id = str(uuid.uuid4())
    
#     video_dir = "fl_tmp"
#     os.makedirs(video_dir, exist_ok=True)
#     video_path = os.path.join(video_dir, f"{task_id}_{file.filename}")
    
#     try:
#         with open(video_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={'error': f'文件保存失败: {str(e)}'}
#         )
    
#     # 初始化任务状态
#     tasks[task_id] = {
#         'state': 'pending',
#         'progress': 0,
#         'result': None,
#         'error': None,
#         'video_path': video_path,
#         'filename': file.filename
#     }
    
#     # 启动后台任务
#     asyncio.create_task(process_video_background(task_id, video_path))
    
#     return JSONResponse({
#         'task_id': task_id,
#         'message': '任务已提交',
#         'filename': file.filename
#     })


# @app.get("/video_handler/status/{task_id}")
# async def get_task_status(task_id: str):
#     '''
#     查询任务处理状态
#     '''
#     if task_id not in tasks:
#         return JSONResponse(
#             status_code=404,
#             content={'error': '任务不存在'}
#         )
    
#     task = tasks[task_id]
#     return JSONResponse({
#         'task_id': task_id,
#         'state': task['state'],
#         'progress': task['progress'],
#         'message': task.get('message', ''),
#         'result': task['result'],
#         'error': task['error']
#     })


# async def process_video_background(task_id: str, video_path: str):
#     '''
#     后台异步处理视频
#     '''
#     start_time = time.time()
    
#     try:
#         # 初始化
#         tasks[task_id]['state'] = 'processing'
#         tasks[task_id]['progress'] = 5
#         tasks[task_id]['message'] = '开始处理视频'
        
#         # 在线程池中执行耗时的同步操作
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(
#             None,
#             process_video_sync,
#             task_id,
#             video_path
#         )
        
#         # 完成
#         tasks[task_id]['state'] = 'completed'
#         tasks[task_id]['progress'] = 100
#         tasks[task_id]['message'] = '处理完成'
#         tasks[task_id]['result'] = result
        
#         end_time = time.time()
#         elapsed_seconds = round(end_time - start_time, 2)
#         print(f"任务 {task_id} 完成, 总耗时: {elapsed_seconds}秒")
        
#     except Exception as e:
#         tasks[task_id]['state'] = 'failed'
#         tasks[task_id]['error'] = str(e)
#         tasks[task_id]['progress'] = 0
#         tasks[task_id]['message'] = '处理失败'
#         print(f"任务 {task_id} 失败: {str(e)}")
    
#     finally:
#         cleanup_temp_files(video_path)


# def process_video_sync(task_id: str, video_path: str):
#     '''
#     同步处理视频（带进度更新）
#     '''
#     ocr_fl = None

#     try:
#         # 包装fl_long_main，添加进度回调
#         def update_progress(progress, message=''):
#             """更新任务进度的回调函数"""
#             if task_id in tasks:
#                 tasks[task_id]['progress'] = progress
#                 tasks[task_id]['message'] = message
#                 print(f"[Task {task_id[:8]}] {progress}% - {message}")
        
#         print(f"任务 {task_id[:8]} - 创建OCR实例...")
#         ocr_fl = PaddleOCR(
#             use_angle_cls=True,
#             lang="ch",
#             show_log=False,
#             use_gpu=True,
#             gpu_mem=500,
#             enable_mkldnn=False,
#             use_tensorrt=False,
#         )
#         print(f"任务 {task_id[:8]} - OCR实例创建成功")
        
#         result = fl_long_main(
#             input_video=video_path,
#             work_dir="fl_tmp_seg",
#             audio_model=audio_model,
#             audio_processor=audio_processor,
#             sam=SAM,
#             class_model=class_model,
#             processor=processor,
#             CLIP_MODEL=CLIP_MODEL,
#             ocr_fl=ocr_fl,
#             sentence_model=sentence_model,
#             ner_pipeline=ner_pipeline,
#             do_search=True,
#             progress_callback=update_progress  # 传入回调
#         )
        
#         return result
        
#     except Exception as e:
#         raise e

#     finally:
#         # 清理OCR实例
#         if ocr_fl is not None:
#             try:
#                 print(f"任务 {task_id[:8]} - 清理OCR实例...")
#                 del ocr_fl
#                 import gc
#                 gc.collect()
#                 try:
#                     import paddle
#                     paddle.device.cuda.empty_cache()
#                 except:
#                     pass
#                 print(f"任务 {task_id[:8]} - OCR实例清理完成")
#             except Exception as e:
#                 print(f"任务 {task_id[:8]} - OCR清理失败: {e}")


# def cleanup_temp_files(video_path: str):
#     '''
#     清理临时文件
#     '''
#     try:
#         if os.path.exists(video_path):
#             os.remove(video_path)
        
#         json_path = os.path.splitext(video_path)[0] + "_analysis.json"
#         if os.path.exists(json_path):
#             os.remove(json_path)
        
#         video_dir = os.path.dirname(video_path)
#         if os.path.exists(video_dir) and not os.listdir(video_dir):
#             os.rmdir(video_dir)
            
#     except Exception as e:
#         print(f"清理临时文件时出错: {e}")


# def main():
#     # 配置Uvicorn运行FastAPI应用
#     config = uvicorn.Config(app, host="0.0.0.0", port=9090, log_level="info")
#     server = uvicorn.Server(config)
#     server.run()


# if __name__ == "__main__":
#     with torch.no_grad():
#         logging.disable(logging.DEBUG)
#         # 主模型
#         CLIP_MODEL = ChineseCLIP()
#         # 切割图像
#         SAM = Segment_generator()
#         SAM = SAM.get_model()
#         tfidf_search = TFIDFSearch(None, False, TFIDF_PATH)
#         search_graph = SearchGraph(TFIDF_PATH)
#         # 图片分类相关模型加载
#         processor = ViTImageProcessor.from_pretrained(CLASS_MODEL_PATH)
#         class_model = vit_model().to('cuda:0')
#         class_model.load_state_dict(torch.load(CLASS_CHECKPOINT))
#         ner_pipeline = modelscope.pipeline(Tasks.named_entity_recognition, model='model/bert_ner', device='cuda:0')
#         # OCR识别，简体：ch，繁体：chinese_cht，showlog: 是否输出日志（3.x移除）
#         # ocr_chinese_cht = PaddleOCR(use_angle_cls=True, lang="chinese_cht", show_log=False, device='cuda:0', gpu_mem=500, enable_mkldnn=False, use_tensorrt=False)
#         # ocr_ch = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, device='cuda:0', gpu_mem=500, enable_mkldnn=False, use_tensorrt=False)
#         # ocr_ko = PaddleOCR(use_angle_cls=True, lang="korean", show_log=False, device='cuda:0', gpu_mem=500, enable_mkldnn=False, use_tensorrt=False)
#         # ocr_fl = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, device='cuda:0', gpu_mem=500, enable_mkldnn=False, use_tensorrt=False)
#         # ocr_chinese_cht = PaddleOCR(use_textline_orientation=True, lang="chinese_cht", device='gpu:0')
#         # ocr_ch = PaddleOCR(use_textline_orientation=True, lang="ch", device='gpu:0')
#         sentence_model = BGEM3FlagModel('model/bge-m3', use_fp16=True, device='cuda:0')
#         # 语音识别模型
#         audio_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
#         audio_model.to('cuda:0')
#         audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL_PATH)

#         text_set = torch.load('data/text_bgem3.pkl')
#         img_text_set = torch.load('data/img_text_bgem3.pkl')
#         video_text_set = torch.load('data/video_text_bgem3_v2.pkl')
#         audio_text_set = torch.load('data/audio_text_bgem3_v2.pkl')
#         elements_set = json.load(open('data/fin_elements_v4.json', 'r', encoding='utf-8'))
#         main()


@app.post("/video_handler")
async def fl_video_handler(file: UploadFile = File(...)):
    '''视频处理 (同步接口，保留向后兼容)'''
    start_time = time.time()
    video_dir = "fl_tmp"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, file.filename)

    ocr_fl = None

    try:
        with open(video_path, "wb") as f:
            f.write(await file.read())

        print("创建OCR实例...")
        ocr_fl = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
            use_gpu=True,
            gpu_mem=300,
            enable_mkldnn=False,
            use_tensorrt=False,
            rec_batch_num=1,
            cls_batch_num=1,
        )
        print("OCR实例创建成功")

        # 处理前清理内存
        cleanup_gpu_memory_enhanced(force=True)

        result = fl_long_main(
            input_video=video_path,
            work_dir="fl_tmp_seg",
            audio_model=audio_model,
            audio_processor=audio_processor,
            sam=SAM,
            class_model=class_model,
            processor=processor,
            CLIP_MODEL=CLIP_MODEL,
            ocr_fl=ocr_fl,
            sentence_model=sentence_model,
            ner_pipeline=ner_pipeline,
            do_search=True
        )

        end_time = time.time()
        elapsed_seconds = round(end_time - start_time, 2)
        print("总耗时: ", elapsed_seconds)
        return result

    except Exception as e:
        print(f"视频处理失败: {e}")
        raise

    finally:
        # 清理OCR实例
        if ocr_fl is not None:
            try:
                print("清理OCR实例...")
                # 先清理内部属性
                if hasattr(ocr_fl, 'detector'):
                    del ocr_fl.detector
                if hasattr(ocr_fl, 'recognizer'):
                    del ocr_fl.recognizer
                if hasattr(ocr_fl, 'cls'):
                    del ocr_fl.cls
                
                del ocr_fl
                # 多次GC清理
                for _ in range(3):
                    gc.collect()
                    try:
                        paddle.device.cuda.empty_cache()
                    except:
                        pass
                print("OCR实例清理完成")
            except Exception as e:
                print(f"OCR清理失败: {e}")
        
        # 强制清理GPU显存
        cleanup_gpu_memory_enhanced(force=True)
        
        # 清理临时文件
        cleanup_temp_files(video_path)


@app.post("/video_handler/submit")
async def submit_video_task(file: UploadFile = File(...)):
    '''提交视频处理任务 (异步接口)'''
    task_id = str(uuid.uuid4())
    
    video_dir = "fl_tmp"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{task_id}_{file.filename}")
    
    try:
        with open(video_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': f'文件保存失败: {str(e)}'}
        )
    
    # 初始化任务状态
    tasks[task_id] = {
        'state': 'pending',
        'progress': 0,
        'result': None,
        'error': None,
        'video_path': video_path,
        'filename': file.filename
    }
    
    # 启动后台任务
    asyncio.create_task(process_video_background(task_id, video_path))
    
    return JSONResponse({
        'task_id': task_id,
        'message': '任务已提交',
        'filename': file.filename
    })


@app.get("/video_handler/status/{task_id}")
async def get_task_status(task_id: str):
    '''查询任务处理状态'''
    if task_id not in tasks:
        return JSONResponse(
            status_code=404,
            content={'error': '任务不存在'}
        )
    
    task = tasks[task_id]
    return JSONResponse({
        'task_id': task_id,
        'state': task['state'],
        'progress': task['progress'],
        'message': task.get('message', ''),
        'result': task['result'],
        'error': task['error']
    })


async def process_video_background(task_id: str, video_path: str):
    '''后台异步处理视频'''
    start_time = time.time()
    
    try:
        # 初始化
        tasks[task_id]['state'] = 'processing'
        tasks[task_id]['progress'] = 5
        tasks[task_id]['message'] = '开始处理视频'
        
        # 在线程池中执行耗时的同步操作
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            process_video_sync,
            task_id,
            video_path
        )
        
        # 完成
        tasks[task_id]['state'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['message'] = '处理完成'
        tasks[task_id]['result'] = result
        
        end_time = time.time()
        elapsed_seconds = round(end_time - start_time, 2)
        print(f"任务 {task_id} 完成, 总耗时: {elapsed_seconds}秒")
        
    except Exception as e:
        tasks[task_id]['state'] = 'failed'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['progress'] = 0
        tasks[task_id]['message'] = '处理失败'
        print(f"任务 {task_id} 失败: {str(e)}")
    
    finally:
        # 强制清理GPU显存
        cleanup_gpu_memory_enhanced(force=True)
        
        # 清理临时文件
        cleanup_temp_files(video_path)


def process_video_sync(task_id: str, video_path: str):
    '''同步处理视频（带进度更新）'''
    ocr_fl = None

    try:
        def update_progress(progress, message=''):
            """更新任务进度的回调函数"""
            if task_id in tasks:
                tasks[task_id]['progress'] = progress
                tasks[task_id]['message'] = message
                print(f"[Task {task_id[:8]}] {progress}% - {message}")
        
        print(f"任务 {task_id[:8]} - 创建OCR实例...")
        ocr_fl = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
            use_gpu=True,
            gpu_mem=500,
            enable_mkldnn=False,
            use_tensorrt=False,
        )
        print(f"任务 {task_id[:8]} - OCR实例创建成功")
        
        result = fl_long_main(
            input_video=video_path,
            work_dir="fl_tmp_seg",
            audio_model=audio_model,
            audio_processor=audio_processor,
            sam=SAM,
            class_model=class_model,
            processor=processor,
            CLIP_MODEL=CLIP_MODEL,
            ocr_fl=ocr_fl,
            sentence_model=sentence_model,
            ner_pipeline=ner_pipeline,
            do_search=True,
            progress_callback=update_progress
        )
        
        return result
        
    except Exception as e:
        raise e

    finally:
        # 清理OCR实例
        if ocr_fl is not None:
            try:
                print(f"任务 {task_id[:8]} - 清理OCR实例...")
                del ocr_fl
                gc.collect()
                try:
                    paddle.device.cuda.empty_cache()
                except:
                    pass
                print(f"任务 {task_id[:8]} - OCR实例清理完成")
            except Exception as e:
                print(f"任务 {task_id[:8]} - OCR清理失败: {e}")
        
        # 强制清理GPU显存
        cleanup_gpu_memory_enhanced(force=True)


def cleanup_temp_files(video_path: str):
    '''清理临时文件'''
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
        
        json_path = os.path.splitext(video_path)[0] + "_analysis.json"
        if os.path.exists(json_path):
            os.remove(json_path)
        
        video_dir = os.path.dirname(video_path)
        if os.path.exists(video_dir) and not os.listdir(video_dir):
            os.rmdir(video_dir)
            
    except Exception as e:
        print(f"清理临时文件时出错: {e}")


def main():
    config = uvicorn.Config(app, host="0.0.0.0", port=9090, log_level="info")
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    with torch.no_grad():
        logging.disable(logging.DEBUG)
        CLIP_MODEL = ChineseCLIP()
        SAM = Segment_generator()
        SAM = SAM.get_model()
        tfidf_search = TFIDFSearch(None, False, TFIDF_PATH)
        search_graph = SearchGraph(TFIDF_PATH)
        processor = ViTImageProcessor.from_pretrained(CLASS_MODEL_PATH)
        class_model = vit_model().to('cuda:0')
        class_model.load_state_dict(torch.load(CLASS_CHECKPOINT))
        ner_pipeline = modelscope.pipeline(Tasks.named_entity_recognition, model='model/bert_ner', device='cuda:0')
        
        sentence_model = BGEM3FlagModel('model/bge-m3', use_fp16=True, device='cuda:0')
        audio_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
        audio_model.to('cuda:0')
        audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL_PATH)

        text_set = torch.load('data/text_bgem3.pkl')
        img_text_set = torch.load('data/img_text_bgem3.pkl')
        video_text_set = torch.load('data/video_text_bgem3_v2.pkl')
        audio_text_set = torch.load('data/audio_text_bgem3_v2.pkl')
        elements_set = json.load(open('data/fin_elements_v4.json', 'r', encoding='utf-8'))
        main()
