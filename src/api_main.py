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

GLOBAL_OCR = None

# 存储任务状态
tasks: Dict[str, dict] = {}

# 任务队列管理
@dataclass
class VideoTask:
    task_id: str
    video_id: str
    video_path: str
    filename: str
    submitted_at: float
    
    table_name: str = "collect_content"
    video_url: Optional[str] = None 
    keyword: Optional[str] = None
    create_time: Optional[str] = None


class VideoTaskQueue:
    def __init__(self, max_size=100):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = False
        self.max_size = max_size
    
    async def put(self, task: VideoTask):
        """添加任务到队列"""
        await self.queue.put(task)
    
    async def get(self) -> VideoTask:
        """从队列获取任务"""
        return await self.queue.get()
    
    def qsize(self) -> int:
        """队列大小"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """队列是否为空"""
        return self.queue.empty()
    
    def full(self) -> bool:
        """队列是否已满"""
        return self.queue.qsize() >= self.max_size


# 全局队列实例
task_queue = VideoTaskQueue(max_size=100)  # 最大50个任务排队
queue_processor_task = None


class VideoTaskRequest(BaseModel):
    """视频任务请求模型"""
    video_id: str
    table_name: str = "collect_content"
    video_url: Optional[str] = None 
    keyword: Optional[str] = None
    create_time: Optional[str] = None


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


@app.on_event("startup")
async def startup_event():
    """启动时启动队列处理器"""
    global queue_processor_task
    queue_processor_task = asyncio.create_task(queue_processor())


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时停止队列处理器"""
    global queue_processor_task
    if queue_processor_task:
        queue_processor_task.cancel()
        try:
            await queue_processor_task
        except asyncio.CancelledError:
            print("队列处理器已停止")


async def queue_processor():
    """队列处理器 - 持续从队列中取出任务处理"""
    print("队列处理器启动...")
    while True:
        task = None
        try:
            # 从队列中获取任务（如果没有任务会等待）
            task = await task_queue.get()
            print(f"开始处理任务: {task.task_id}, 视频ID: {task.video_id}")
            
            # 更新任务状态
            tasks[task.task_id]['state'] = 'processing'
            tasks[task.task_id]['progress'] = 5
            tasks[task.task_id]['message'] = '开始处理视频'
            tasks[task.task_id]['started_at'] = time.time()
            
            # 处理任务
            await process_single_task(task)
            
            # 标记任务完成
            task_queue.queue.task_done()
            print(f"任务完成: {task.task_id}, 视频ID: {task.video_id}")
            
        except asyncio.CancelledError:
            print("队列处理器被取消")
            break
        except Exception as e:
            print(f"队列处理器异常: {e}")
            # 记录错误但继续处理下一个任务
            if task and task.task_id in tasks:
                tasks[task.task_id]['state'] = 'failed'
                tasks[task.task_id]['error'] = str(e)
                tasks[task.task_id]['progress'] = 0
                tasks[task.task_id]['message'] = '处理失败'
                tasks[task.task_id]['completed_at'] = time.time()

            if task:
                try:
                    task_queue.queue.task_done()
                except ValueError:
                    pass


def get_queue_position(target_task_id: str) -> int:
    """
    计算任务在等待队列中的位置 (从1开始)
    原理：统计所有状态为 pending 的任务，按提交时间排序，找到当前任务的索引
    """
    try:
        # 1. 筛选出所有待处理的任务
        pending_list = []
        for t_id, t_info in tasks.items():
            if t_info.get('state') == 'pending':
                pending_list.append({
                    'task_id': t_id,
                    'submitted_at': t_info.get('submitted_at', 0)
                })
        
        # 2. 按提交时间先后排序 (FIFO)
        pending_list.sort(key=lambda x: x['submitted_at'])
        
        # 3. 查找目标任务的位置
        for idx, item in enumerate(pending_list):
            if item['task_id'] == target_task_id:
                return idx + 1
                
        return 0 # 如果没找到（可能已经开始处理或已完成）
    except Exception as e:
        print(f"计算队列位置出错: {e}")
        return 0


def estimate_wait_time(position: int) -> int:
    """
    根据队列位置估算等待时间 (返回秒数)
    """
    if position <= 0:
        return 
        
    AVG_PROCESS_TIME_PER_VIDEO = 60  # 秒
    total_seconds = position * AVG_PROCESS_TIME_PER_VIDEO
    
    return total_seconds


async def process_single_task(task: VideoTask):
    """处理单个任务"""
    start_time = time.time()
    
    try:
        # 在线程池中执行耗时的同步操作
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            process_video_sync,
            task.task_id,
            task.video_path,
            task  # 传递完整的任务对象，包含元数据
        )
        
        # 任务完成
        tasks[task.task_id]['state'] = 'completed'
        tasks[task.task_id]['progress'] = 100
        tasks[task.task_id]['message'] = '处理完成'
        tasks[task.task_id]['result'] = result
        tasks[task.task_id]['completed_at'] = time.time()
        
        # 在结果中添加元数据信息，便于客户端使用
        if isinstance(result, dict):
            result['metadata'] = {
                'table_name': task.table_name,
                'video_id': task.video_id,
                'video_url': task.video_url,
                'keyword': task.keyword,
                'create_time': task.create_time
            }
        
        end_time = time.time()
        elapsed_seconds = round(end_time - start_time, 2)
        print(f"任务 {task.task_id} 完成, 视频ID: {task.video_id}, 总耗时: {elapsed_seconds}秒")
        
    except Exception as e:
        # 任务失败
        tasks[task.task_id]['state'] = 'failed'
        tasks[task.task_id]['error'] = str(e)
        tasks[task.task_id]['progress'] = 0
        tasks[task.task_id]['message'] = '处理失败'
        tasks[task.task_id]['completed_at'] = time.time()
        print(f"任务 {task.task_id} 失败, 视频ID: {task.video_id}: {str(e)}")
    
    finally:
        # 强制清理GPU显存
        cleanup_gpu_memory_enhanced(force=True)
        # 清理临时文件
        cleanup_temp_files(task.video_path)


@app.post("/video_handler/submit")
async def submit_video_task(
    file: UploadFile = File(...),
    task_request: str = Form(...)
):
    '''提交视频处理任务到队列（支持元数据）'''
    # 检查队列是否已满
    if task_queue.full():
        return JSONResponse(
            status_code=429,
            content={
                'error': '任务队列已满',
                'message': f'当前队列中有{task_queue.qsize()}个任务等待处理，请稍后再提交',
                'max_queue_size': task_queue.max_size
            }
        )
    
    # 解析任务请求
    try:
        task_data = json.loads(task_request)
        task_request_obj = VideoTaskRequest(**task_data)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={'error': f'任务请求格式错误: {str(e)}'}
        )
    
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
    
    # 创建任务对象（包含元数据）
    task = VideoTask(
        task_id=task_id,
        table_name=task_request_obj.table_name or "collect_content",
        video_id=task_request_obj.video_id,
        video_path=video_path,
        filename=file.filename,
        submitted_at=time.time(),
        video_url=task_request_obj.video_url,
        keyword=task_request_obj.keyword,
        create_time=task_request_obj.create_time
    )
    
    # 初始化任务状态（包含元数据）
    tasks[task_id] = {
        'state': 'pending',
        'progress': 0,
        'result': None,
        'error': None,
        'video_path': video_path,
        'filename': file.filename,
        'submitted_at': task.submitted_at,
        'queue_position': task_queue.qsize() + 1,
        # 存储元数据
        'table_name': task.table_name,
        'video_id': task.video_id,
        'video_url': task.video_url,
        'keyword': task.keyword,
        'create_time': task.create_time
    }
    
    # 将任务加入队列
    await task_queue.put(task)
    
    # 更新队列位置信息
    for t_id, task_info in tasks.items():
        if task_info['state'] == 'pending':
            # 重新计算队列位置
            task_info['queue_position'] = get_queue_position(t_id)
    
    return JSONResponse({
        'task_id': task_id,
        'message': '任务已加入队列',
        'filename': file.filename,
        'video_id': task.video_id,
        'queue_position': tasks[task_id]['queue_position'],
        'estimated_wait_time': estimate_wait_time(tasks[task_id]['queue_position']),
        'queue_size': task_queue.qsize()
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
    response_data = {
        'task_id': task_id,
        'state': task['state'],
        'progress': task['progress'],
        'message': task.get('message', ''),
        'filename': task['filename'],
        'submitted_at': task.get('submitted_at'),
        'started_at': task.get('started_at'),
        'completed_at': task.get('completed_at'),
        # 返回元数据
        'table_name': task.get('table_name', 'collect_content'),
        'video_id': task.get('video_id'),
        'video_url': task.get('video_url'),
        'keyword': task.get('keyword'),
        'create_time': task.get('create_time')
    }
    
    # 根据状态添加额外信息
    if task['state'] == 'pending':
        response_data['queue_position'] = task.get('queue_position', 0)
        response_data['estimated_wait_time'] = estimate_wait_time(task.get('queue_position', 0))
    elif task['state'] == 'completed':
        response_data['result'] = task['result']
    elif task['state'] == 'failed':
        response_data['error'] = task['error']
    
    return JSONResponse(response_data)


@app.get("/video_handler/queue/status")
async def get_queue_status():
    '''获取队列状态'''
    status_counts = {
        'pending': 0,
        'processing': 0,
        'completed': 0,
        'failed': 0,
        'total': len(tasks)
    }
    
    current_task = None
    pending_tasks = []
    
    for task_id, task in tasks.items():
        status_counts[task['state']] += 1
        
        if task['state'] == 'processing':
            current_task = {
                'task_id': task_id,
                'filename': task['filename'],
                'video_id': task.get('video_id'),
                'progress': task['progress'],
                'message': task.get('message', ''),
                'started_at': task.get('started_at')
            }
        elif task['state'] == 'pending':
            pending_tasks.append({
                'task_id': task_id,
                'filename': task['filename'],
                'video_id': task.get('video_id'),
                'keyword': task.get('keyword'),
                'submitted_at': task.get('submitted_at'),
                'queue_position': task.get('queue_position', 0),
                'estimated_wait_time': estimate_wait_time(task.get('queue_position', 0))
            })
    
    # 按提交时间排序待处理任务
    pending_tasks.sort(key=lambda x: x.get('submitted_at', 0))
    
    return JSONResponse({
        'status_counts': status_counts,
        'queue_info': {
            'current_size': task_queue.qsize(),
            'max_size': task_queue.max_size,
            'is_full': task_queue.full(),
            'is_empty': task_queue.empty()
        },
        'current_task': current_task,
        'pending_tasks': pending_tasks[:10],
        'server_time': time.time()
    })


def process_video_sync(task_id: str, video_path: str, task: VideoTask = None):
    '''同步处理视频（带进度更新和元数据支持）'''
    try:
        def update_progress(progress, message=''):
            """更新任务进度的回调函数"""
            if task_id in tasks:
                tasks[task_id]['progress'] = progress
                tasks[task_id]['message'] = message
                print(f"[Task {task_id[:8]}] VideoID: {task.video_id if task else 'N/A'}, {progress}% - {message}")
                
        # 在处理前强制清理一次
        cleanup_gpu_memory_enhanced(force=True)
        
        # 调用处理函数，可以传递元数据
        result = fl_long_main(
            input_video=video_path,
            work_dir="fl_tmp_seg",
            audio_model=audio_model,
            audio_processor=audio_processor,
            sam=SAM,
            class_model=class_model,
            processor=processor,
            CLIP_MODEL=CLIP_MODEL,
            ocr_fl=GLOBAL_OCR,
            sentence_model=sentence_model,
            ner_pipeline=ner_pipeline,
            do_search=True,
            progress_callback=update_progress
        )
        
        # 在结果中添加元数据
        if task and isinstance(result, dict):
            result['table_name'] = task.table_name
            result['video_id'] = task.video_id
            result['video_url'] = task.video_url
            result['keyword'] = task.keyword
            result['create_time'] = task.create_time
        
        return result
        
    except Exception as e:
        raise e

    finally:        
        # 最终强制清理
        cleanup_gpu_memory_enhanced(force=True)


@app.post("/video_handler")
async def fl_video_handler(file: UploadFile = File(...)):
    '''视频处理 (同步接口，保留向后兼容)'''
    start_time = time.time()
    video_dir = "fl_tmp"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, file.filename)

    try:
        with open(video_path, "wb") as f:
            f.write(await file.read())

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
            ocr_fl=GLOBAL_OCR,
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
        # 强制清理GPU显存
        cleanup_gpu_memory_enhanced(force=True)
        # 清理临时文件
        cleanup_temp_files(video_path)


def cleanup_temp_files(video_path: str):
    '''清理临时文件'''
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"已删除临时视频文件: {video_path}")
        
        # 清理可能的分析结果文件
        json_path = os.path.splitext(video_path)[0] + "_analysis.json"
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"已删除分析结果文件: {json_path}")
        
        # 清理临时目录（如果为空）
        video_dir = os.path.dirname(video_path)
        if os.path.exists(video_dir) and not os.listdir(video_dir):
            os.rmdir(video_dir)
            print(f"已删除空临时目录: {video_dir}")
            
    except Exception as e:
        print(f"清理临时文件时出错: {e}")


def main():
    config = uvicorn.Config(app, host="0.0.0.0", port=9090, log_level="info")
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    with torch.no_grad():
        logging.disable(logging.DEBUG)

        print("正在加载模型资源...")

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

        print("正在初始化全局 PaddleOCR...")
        GLOBAL_OCR = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
            use_gpu=True,
            gpu_mem=500,
            enable_mkldnn=False,
            use_tensorrt=False,
            rec_batch_num=1,
            cls_batch_num=1,
        )
        print("全局 PaddleOCR 初始化完成")

        text_set = torch.load('data/text_bgem3.pkl')
        img_text_set = torch.load('data/img_text_bgem3.pkl')
        video_text_set = torch.load('data/video_text_bgem3_v2.pkl')
        audio_text_set = torch.load('data/audio_text_bgem3_v2.pkl')
        elements_set = json.load(open('data/fin_elements_v4.json', 'r', encoding='utf-8'))
        main()
