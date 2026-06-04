import cv2
import gc
import glob
import json
import os
import paddle
import re
import shutil
import subprocess
import time
import torch
import traceback
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from audio_handler import transcribe_long_audio
from frame_utils import check_repetition
from img_helpers import img_recg
from my_utils import call_translator, call_qwen, element_filter, call_qwen_smart, seconds_to_hms
from segment_helpers import get_segment_images, get_imgs
from video_handler import read_mov, dpc_main, chinese_ner
from bge_m3_helper import search_text


# def cleanup_gpu_memory_enhanced(force=False):
#     """
#     增强的GPU显存清理函数
    
#     Args:
#         force: 是否强制清理（包括缓存的模型）
#     """
#     if not torch.cuda.is_available():
#         return
    
#     try:
#         gc.collect()
#         torch.cuda.empty_cache()
        
#         if force:
#             torch.cuda.synchronize()
#             torch.cuda.reset_peak_memory_stats()
            
#     except Exception as e:
#         print(f"GPU清理失败: {e}")


def cleanup_gpu_memory_enhanced(force=False):
    """
    增强的GPU显存清理函数（针对PyTorch和PaddlePaddle）
    
    Args:
        force: 是否强制清理（包括多次循环清理）
    """
    print("开始GPU内存清理...")
    
    try:
        # 第一步：Python垃圾回收
        collected = gc.collect()
        print(f"  Python GC完成，回收对象: {collected}")
        
        # 第二步：清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if force:
                torch.cuda.reset_peak_memory_stats()
            print("  PyTorch GPU缓存清理完成")
        
        # 第三步：清理PaddlePaddle缓存
        try:
            paddle.device.cuda.empty_cache()
            print("  PaddlePaddle GPU缓存清理完成")
        except Exception as e:
            print(f"  PaddlePaddle清理警告: {e}")
        
        # 第四步：强制模式下的额外清理
        if force:
            # 多次GC循环确保彻底清理
            for i in range(3):
                collected = gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    paddle.device.cuda.empty_cache()
                except:
                    pass
                print(f"  第{i+1}次强制清理循环完成，回收对象: {collected}")
        
        # 第五步：报告内存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3     # GB
            print(f"  内存状态: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")
        
        print("GPU内存清理完成")
        
    except Exception as e:
        print(f"GPU清理失败: {e}")


def extract_audio(video_path, audio_path, sr=16000, channels=1):
    '''从视频中抽取音频并保存为wav'''
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", str(channels),
        "-ar", str(sr),
        "-f", "wav",
        audio_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    
    if result.returncode != 0 or not os.path.exists(audio_path):
        raise Exception(f"音频提取失败: {result.stderr.decode('utf-8', errors='ignore')}")
    
    if os.path.getsize(audio_path) == 0:
        raise Exception("提取的音频文件为空")


def extract_full_audio(video_path, sr=16000, channels=1, audio_model=None, audio_processor=None):
    '''抽取视频的完整音频，并一次性长转写'''
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    audio_path = os.path.splitext(video_path)[0] + "_full.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", str(channels),
        "-ar", str(sr),
        "-f", "wav",
        audio_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    
    if result.returncode != 0 or not os.path.exists(audio_path):
        raise Exception(f"音频提取失败: {result.stderr.decode('utf-8', errors='ignore')}")
    
    if os.path.getsize(audio_path) == 0:
        raise Exception("提取的音频文件为空")
    
    with torch.no_grad():
        asr_result = transcribe_long_audio(audio_path, model=audio_model, processor=audio_processor)
    
    if not asr_result:
        print("警告: ASR转写结果为空")
        asr_result = []
    
    return audio_path, asr_result


def slice_video(input_path, output_dir, segment_seconds=60):
    '''将长视频按segment_seconds切片'''
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"视频文件不存在: {input_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", input_path,
        "-c", "copy", "-map", "0",
        "-f", "segment", "-segment_time", str(segment_seconds),
        os.path.join(output_dir, "segment_%03d.mp4")
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    
    if result.returncode != 0:
        raise Exception(f"视频切片失败: {result.stderr.decode('utf-8', errors='ignore')}")
    
    seg_files = glob.glob(os.path.join(output_dir, "segment_*.mp4"))
    if not seg_files:
        raise Exception("视频切片失败: 未生成任何切片文件")
    
    print(f"切片完成，共生成 {len(seg_files)} 个片段")


def slice_transcript(full_asr, start_s, end_s):
    '''从整片转写中获取[start_s, end_s)的文本'''
    if not full_asr:
        return ""
    
    if isinstance(full_asr, list):
        parts = []
        for seg in full_asr:
            if not seg or not isinstance(seg, dict):
                continue
            
            st = float(seg.get("start", 0))
            ed = float(seg.get("end", 0))
            tx = (seg.get("text") or "").strip()
            
            if not tx:
                continue
            
            center = (st + ed) / 2
            
            if start_s <= center < end_s:
                parts.append(tx)
        
        return " ".join(parts)
    
    if isinstance(full_asr, str):
        return full_asr.strip()
    
    return ""


def find_overlap(text1, text2, min_len=10):
    """找出 text1 结尾和 text2 开头的重叠部分"""
    if not text1 or not text2:
        return 0
    
    max_overlap = min(len(text1), len(text2), 100)
    
    for length in range(max_overlap, min_len - 1, -1):
        if text1[-length:] == text2[:length]:
            return length
    return 0


def extract_useful_ocr(ocr_text, min_len=10, max_len=150):
    """从 OCR 文本中提取有用的句子"""
    if not ocr_text or not isinstance(ocr_text, str):
        return []
    
    sentences = re.split('[。！？\n]+', ocr_text)
    
    useful = []
    for sent in sentences:
        sent = sent.strip()
        
        if (min_len < len(sent) < max_len and
            re.search(r'[\u4e00-\u9fff]', sent) and
            not re.match(r'^[0-9\-/\.]+$', sent) and
            not re.match(r'^http', sent) and
            not re.match(r'^[@#]', sent)):
            
            sent = re.sub(r'图片来源[：:].+', '', sent)
            sent = re.sub(r'画面[：:].+', '', sent)
            sent = re.sub(r'网址[：:].+', '', sent)
            sent = re.sub(r'\s+', '', sent)
            
            if len(sent) > min_len:
                useful.append(sent)
    
    return useful


def is_substring_in_text(substring, text):
    """检查 substring 是否已经在 text 中出现过"""
    if not substring or not text:
        return False
    
    if substring in text:
        return True
    
    threshold = len(substring) * 0.8
    for i in range(len(text) - len(substring) + 1):
        chunk = text[i:i+len(substring)]
        matches = sum(1 for a, b in zip(substring, chunk) if a == b)
        if matches >= threshold:
            return True
    
    return False


def get_text_elements(query_sentence, ner_pipeline):
    """识别文本中的元素（带显存清理）"""
    if not query_sentence or not isinstance(query_sentence, str):
        return {
            "elements_response": "",
            "elements_names": {},
            "elements_neighbors": {}
        }
    
    try:
        with open('data/fin_elements_v4.json', 'r', encoding='utf-8') as f:
            elements_set = json.load(f)
    except Exception as e:
        print(f"警告: 无法加载元素集合文件: {e}")
        elements_set = []
    
    try:
        elements_response = call_qwen(query_sentence)
        if not elements_response:
            elements_response = ""
        else:
            elements_response = str(elements_response)[:500]
    except Exception as e:
        print(f"警告: Qwen调用失败: {e}")
        elements_response = ""
        
    try:
        with torch.no_grad():
            names = chinese_ner(elements_response, ner_pipeline=ner_pipeline)
        if not names or not isinstance(names, dict):
            names = {}
    except Exception as e:
        print(f"警告: NER识别失败: {e}")
        names = {}
        
    mp = {'PER': ["Politicians"], 'LOC': ["Nation"], 'ORG': ["Orgnization"]}
    graph_neighbors = {}
    
    for key in names.keys():
        if key not in mp:
            continue
        
        skind = mp[key]
        for name in names[key]:
            flag = False
            for element_eve in elements_set:
                if not element_eve:
                    continue
                for element_eve2 in element_eve:
                    if not element_eve2 or not isinstance(element_eve2, dict):
                        continue
                    if name in element_eve2.get('name', '') and skind == element_eve2.get('labels'):
                        graph_neighbors[name] = element_eve
                        flag = True
                        break
                if flag:
                    break
    
    result = {
        "elements_response": elements_response,
        "elements_names": names,
        "elements_neighbors": graph_neighbors
    }
    
    del elements_set
    torch.cuda.empty_cache()
    gc.collect()
    
    return result


def fl_video_main(
    video_path,
    audio_model=None,
    audio_processor=None,
    sam=None,
    class_model=None,
    processor=None,
    CLIP_MODEL=None,
    ocr_fl=None,
    sentence_model=None,
    ner_pipeline=None,
    prefilled_audio_fl=None,
    prefilled_audio_zh=None,
    segment_start_time=0.0,
    progress_callback=None,
    progress_range=(0, 100)
):
    """视频处理主函数（带显存清理）"""
    def update_progress(local_progress, message=''):
        """将局部进度映射到全局进度范围"""
        if progress_callback:
            start, end = progress_range
            actual_progress = start + (end - start) * local_progress / 100
            try:
                progress_callback(int(actual_progress), message)
            except Exception as e:
                print(f"进度回调失败: {e}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    try:
        # ========== 阶段 1: 音频处理 ==========
        update_progress(5, '正在提取音频...')
        
        if prefilled_audio_fl is not None:
            fl_audio_text = prefilled_audio_fl if prefilled_audio_fl else []
            zh_audio_text = prefilled_audio_zh if prefilled_audio_zh else ""
            if not zh_audio_text and fl_audio_text:
                 text_content = " ".join(seg.get("文本", "") for seg in fl_audio_text if seg.get("文本"))
                 if text_content:
                     zh_audio_text = call_translator(text_content)
            update_progress(15, '使用预填充音频')
        else:
            update_progress(2, '正在提取音频...')
            audio_path = os.path.splitext(video_path)[0] + "_tmp.wav"
            try:
                extract_audio(video_path, audio_path)
                update_progress(5, '正在转录音频...')
                
                with torch.no_grad():
                    fl_audio_text = transcribe_long_audio(audio_path, model=audio_model, processor=audio_processor)
                update_progress(10, '正在翻译音频...')
                
                if isinstance(fl_audio_text, list):
                    text_for_translation = " ".join(seg.get("text", "") for seg in fl_audio_text if seg.get("text"))
                elif isinstance(fl_audio_text, str):
                    text_for_translation = fl_audio_text
                else:
                    text_for_translation = ""
                
                zh_audio_text = call_translator(text_for_translation) if text_for_translation else ""
                update_progress(15, '音频处理完成')
            
            except Exception as e:
                print(f"警告: 音频处理失败: {e}")
                fl_audio_text = []
                zh_audio_text = ""
                update_progress(15, '音频处理失败，跳过')
            finally:
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except:
                        pass
        
        print("full_asr:", fl_audio_text)
        print("翻译full_asr:", zh_audio_text)

        cleanup_gpu_memory_enhanced()
        
        # ========== 阶段 2: 视频加载 ==========
        update_progress(15, '正在加载视频...')
        
        try:
            g = read_mov(video_path)
        except Exception as e:
            raise Exception(f"视频加载失败: {e}")
        
        frames = []
        frame_txts_fl = [[]]
        frame_persons = []
        frame_nations = []
        frame_others = []
        element_dicts = []
        key_words = []
        num = -1
        fps = None
        
        cleanup_gpu_memory_enhanced()
        
        # ========== 阶段 3: OCR处理 ==========
        update_progress(20, '正在提取视频帧...')
        
        last_update_progress = 20

        with torch.no_grad():
            while True:
                try:
                    flag, frame, cfps = next(g)
                    num += 1
                    if fps is None:
                        fps = cfps if cfps > 0 else 30
                    if not flag:
                        break
                    if num % fps != 3:
                        continue
                    
                    if frame is None or frame.size == 0:
                        continue
                    
                    frames.append(frame)
                    
                    if num % 10 == 0 and num > 0:
                        current_progress = min(20 + int(30 * num / 300), 50)
                        if current_progress > last_update_progress:
                            update_progress(current_progress, f'已处理 {len(frames)} 帧')
                            last_update_progress = current_progress
                    
                    try:
                        if ocr_fl is None:
                            continue
                        
                        results_fl = ocr_fl.ocr(frame, cls=True)
                        if not results_fl or not results_fl[0]:
                            continue
                        
                        results_fl = results_fl[0]
                        tmp_results_fl = []
                        for result in results_fl:
                            if not result or len(result) < 2:
                                continue
                            prob = result[1][1] if len(result[1]) > 1 else 0
                            if prob >= 0.9:
                                tmp_results_fl.append(result[1][0])
                        
                        if len(tmp_results_fl) == 0 or check_repetition(frame_txts_fl[-1], tmp_results_fl):
                            continue
                        frame_txts_fl.append((tmp_results_fl, num))
                        
                        if num % 5 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                    
                    except Exception as e:
                        print(f"OCR处理第{num}帧时出错: {e}")
                        continue
                except StopIteration:
                    break
                except Exception as e:
                    print(f"处理第{num}帧时出错: {e}")
                    break
        
        update_progress(50, f'帧提取完成，共 {len(frames)} 帧')
        
        if not frames:
            print("警告: 未提取到任何视频帧")
            return [], {
                "elements_response": "",
                "elements_names": {},
                "elements_neighbors": {}
            }, {
                "audio_text_fl": fl_audio_text,
                "audio_text_zh": zh_audio_text,
                "video_text_fl": "",
                "video_text_zh": ""
            }
        
        # ========== 阶段 4: DPC聚类 ==========
        update_progress(50, '正在进行关键帧聚类...')
        
        try:
            with torch.no_grad():
                key_frames_ids = dpc_main(frames, output_path='', isfile=False)
            
            if not key_frames_ids or len(key_frames_ids) == 0:
                print("警告: DPC聚类未返回任何关键帧，使用均匀采样")
                step = max(1, len(frames) // 10)
                key_frames_ids = list(range(0, len(frames), step))
        except Exception as e:
            print(f"警告: DPC聚类失败: {e}，使用均匀采样")
            step = max(1, len(frames) // 10)
            key_frames_ids = list(range(0, len(frames), step))
        
        if not key_frames_ids:
            key_frames_ids = [0] if frames else []
        
        key_frames = [(frames[i], i) for i in key_frames_ids if i < len(frames)]
        
        if not key_frames:
            print("警告: 没有有效的关键帧")
            return [], {
                "elements_response": "",
                "elements_names": {},
                "elements_neighbors": {}
            }, {
                "audio_text_fl": fl_audio_text,
                "audio_text_zh": zh_audio_text,
                "video_text_fl": "",
                "video_text_zh": ""
            }
        
        update_progress(55, f'检测到 {len(key_frames)} 个关键帧')
        
        try:
            if os.path.exists(f'tmp/key_frames'):
                shutil.rmtree(f'tmp/key_frames')
            os.makedirs(f'tmp/key_frames', exist_ok=True)
            for key_frame in key_frames:
                cv2.imwrite(f'tmp/key_frames/{key_frame[1]}.jpg', key_frame[0])
        except Exception as e:
            print(f"警告: 保存关键帧失败: {e}")
        
        update_progress(60, '关键帧保存完成')
        cleanup_gpu_memory_enhanced()
        
        # ========== 阶段 5: SAM识别 ==========
        update_progress(60, '开始分析关键帧...')
        
        try:
            if os.path.exists(f'tmp/segment_frames'):
                shutil.rmtree(f'tmp/segment_frames')
            os.makedirs(f'tmp/segment_frames', exist_ok=True)
        except Exception as e:
            print(f"警告: 创建临时目录失败: {e}")
        
        total_key_frames = len(key_frames)
        
        with torch.no_grad():
            for idx, (key_frame, key_frames_id) in enumerate(key_frames):
                frame_progress = 60 + int(30 * idx / total_key_frames)
                update_progress(frame_progress, f'正在分析关键帧 {idx+1}/{total_key_frames}')

                print(f"处理关键帧 {idx+1}/{total_key_frames}")

                try:
                    frame_dir = f'tmp/segment_frames/{key_frames_id}'
                    os.makedirs(frame_dir, exist_ok=True)
                    
                    if sam is not None:
                        get_segment_images(sam, key_frame, frame_dir)
                    
                    seg_pics, fs = get_imgs(frame_dir)
                    
                    if not seg_pics:
                        continue
                    
                    local_timestamp = key_frames_id / fps if fps > 0 else 0
                    global_timestamp = segment_start_time + local_timestamp
                    
                    for seg_pic in seg_pics:
                        try:
                            element_dict, keyword = img_recg(processor, class_model, CLIP_MODEL, seg_pic)
                            
                            if not element_dict:
                                continue
                            
                            element_dict['frame_idx'] = key_frames_id
                            element_dict['local_timestamp'] = seconds_to_hms(local_timestamp)
                            element_dict['global_timestamp'] = seconds_to_hms(global_timestamp)
                            
                            elem_class = element_dict.get('class', 3)
                            if elem_class == 0:
                                frame_persons.append(element_dict.get('elements', []))
                            elif elem_class == 1:
                                frame_nations.append(element_dict.get('elements', []))
                            elif elem_class == 2:
                                frame_others.append(element_dict.get('elements', []))
                            
                            if elem_class != 3:
                                element_dicts.append(element_dict)
                                if keyword:
                                    key_words.append(keyword)
                        except Exception as e:
                            print(f"  警告: 识别子图失败: {e}")
                            continue
                    
                    if idx % 5 == 0:
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"  警告: 处理关键帧{idx+1}失败: {e}")
                    continue
        
        update_progress(90, '关键帧分析完成')
        cleanup_gpu_memory_enhanced()
        
        # ========== 阶段 6: 文本处理 ==========
        update_progress(90, '正在处理文本...')
        
        ans_lst = []
        if len(frame_txts_fl) > 1:
            ans_lst = frame_txts_fl[1][0] if isinstance(frame_txts_fl[1], tuple) else []
            for i in range(2, len(frame_txts_fl)):
                tmp_lst = frame_txts_fl[i][0] if isinstance(frame_txts_fl[i], tuple) else []
                for eve in tmp_lst:
                    if eve not in ans_lst:
                        ans_lst.append(eve)
        
        video_text_fl = ' '.join(ans_lst)[:1000] if ans_lst else ""
        update_progress(92, '正在翻译文本...')
        
        video_text_zh = call_translator(video_text_fl) if video_text_fl else ""
        video_text_zh = video_text_zh[:1000] if video_text_zh else ""
        
        element_dicts = element_filter(element_dicts) if element_dicts else []
        print("zh_ocr_lst: ", video_text_zh)

        text_dicts = {
            "audio_text_fl": fl_audio_text,
            "audio_text_zh": zh_audio_text,
            "video_text_fl": video_text_fl,
            "video_text_zh": video_text_zh
        }
        print("visual_elements: ", element_dicts)
        
        update_progress(95, '正在提取文本实体...')
        
        with torch.no_grad():
            text_elements = get_text_elements(video_text_zh, ner_pipeline)
        print("text_elements: ", text_elements)

        cleanup_gpu_memory_enhanced()
        
        # ========== 阶段 7: 最终清理 ==========
        update_progress(98, '处理完成')
        cleanup_gpu_memory_enhanced(force=True)
        update_progress(100, '全部完成')
        
        return element_dicts, text_elements, text_dicts

    except Exception as e:
        print(f"处理失败: {e}")
        traceback.print_exc()
        cleanup_gpu_memory_enhanced(force=True)
        raise
    finally:
        if 'frames' in locals():
            del frames
        cleanup_gpu_memory_enhanced(force=True)


def process_segments(output_dir, full_asr_segments, segment_seconds=60, progress_callback=None, **kwargs):
    """按顺序处理每个切片"""
    def update_progress(progress, message=''):
        if progress_callback:
            try:
                progress_callback(int(progress), message)
            except Exception as e:
                print(f"进度回调失败: {e}")
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"切片目录不存在: {output_dir}")
    
    segments_results = []
    seg_files = sorted(glob.glob(os.path.join(output_dir, "segment_*.mp4")))
    
    if not seg_files:
        raise Exception(f"未找到任何视频切片文件: {output_dir}")
    
    total_segments = len(seg_files)
    
    for i, seg_file in enumerate(seg_files):
        progress_per_segment = 65.0 / total_segments
        segment_start_progress = 25 + (i * progress_per_segment)
        segment_end_progress = segment_start_progress + progress_per_segment
        
        update_progress(
            int(segment_start_progress),
            f'开始处理片段 {i+1}/{total_segments}'
        )
        
        if not os.path.exists(seg_file):
            print(f"警告: 切片文件不存在，跳过: {seg_file}")
            update_progress(
                int(segment_end_progress),
                f'片段 {i+1} 文件不存在，已跳过'
            )
            continue
        
        start_sec = i * segment_seconds
        end_sec = (i + 1) * segment_seconds
        
        update_progress(
            int(segment_start_progress + progress_per_segment * 0.05),
            f'片段 {i+1}: 正在处理音频...'
        )
        
        fl_audio_text = slice_transcript(full_asr_segments, start_sec, end_sec)
        zh_audio_text = call_translator(fl_audio_text) if fl_audio_text else ""
        
        try:
            element_dicts, text_elements, text_dicts = fl_video_main(
                seg_file,
                prefilled_audio_fl=fl_audio_text,
                prefilled_audio_zh=zh_audio_text,
                segment_start_time=start_sec,
                progress_callback=progress_callback,
                progress_range=(
                    int(segment_start_progress + progress_per_segment * 0.05),
                    int(segment_end_progress)
                ),
                **kwargs
            )
            
            update_progress(
                int(segment_end_progress),
                f'片段 {i+1}/{total_segments} 处理完成'
            )
        
        except Exception as e:
            print(f"警告: 处理片段{i}失败: {e}")
            
            update_progress(
                int(segment_end_progress),
                f'片段 {i+1} 处理失败: {str(e)[:50]}...'
            )
            
            element_dicts = []
            text_elements = {
                "elements_response": "",
                "elements_names": {},
                "elements_neighbors": {}
            }
            text_dicts = {
                "audio_text_fl": fl_audio_text,
                "audio_text_zh": zh_audio_text,
                "video_text_fl": "",
                "video_text_zh": ""
            }
        
        segments_results.append({
            "segment_id": i,
            "segment_file": os.path.basename(seg_file),
            "start_time_sec": start_sec,
            "end_time_sec": end_sec,
            "element_dicts": element_dicts if element_dicts else [],
            "text_elements": text_elements if text_elements else {},
            "text_dicts": text_dicts if text_dicts else {}
        })
    
    if not segments_results:
        raise Exception("所有视频片段处理均失败")
    
    update_progress(90, f'所有 {total_segments} 个片段处理完成')
    
    return segments_results


def convert_local_path_to_url(local_path, base_url="http://117.50.191.107:6077"):
    """将本地路径转换为HTTP URL"""
    local_path = local_path.replace('\\', '/')
    
    if 'candidate_set' in local_path:
        relative_path = local_path.split('candidate_set', 1)[1].lstrip('/')
        return f"{base_url}/candidate_set/{relative_path}"
    else:
        return f"{base_url}/{local_path}"


def query_search(query_sentence, use_url=True, base_url="http://117.50.191.107:6077"):
    """根据query检索库中数据"""
    text_set = torch.load('data/text_bgem3.pkl')
    img_text_set = torch.load('data/img_text_bgem3.pkl')
    video_text_set = torch.load('data/video_text_bgem3_v2.pkl')
    
    sentence_model = BGEM3FlagModel('model/bge-m3', use_fp16=True, device='cuda:0')
    
    with torch.no_grad():
        text_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=text_set)
        texts = [eve[1] for eve in text_ans]
    
        img_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=img_text_set)
        if use_url:
            images = [
                {
                    'image': convert_local_path_to_url(
                        os.path.join(os.getcwd(), 'candidate_set/images', eve[1]),
                        base_url
                    ), 
                    'text': eve[2]
                } 
                for eve in img_ans
            ]
        else:
            images = [
                {
                    'image': os.path.join(os.getcwd(), 'candidate_set/images', eve[1]), 
                    'text': eve[2]
                } 
                for eve in img_ans
            ]
        
        videos_ans = search_text(query=query_sentence, model=sentence_model, candidate_set=video_text_set)
        if use_url:
            videos = [
                {
                    'video': convert_local_path_to_url(
                        os.path.join(os.getcwd(), 'candidate_set/videos', eve[1]),
                        base_url
                    ), 
                    'text': eve[2]
                } 
                for eve in videos_ans
            ]
        else:
            videos = [
                {
                    'video': os.path.join(os.getcwd(), 'candidate_set/videos', eve[1]), 
                    'text': eve[2]
                } 
                for eve in videos_ans
            ]

    del sentence_model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'texts': texts, 
        'images': images,
        'videos': videos
    }


def has_extreme_repetition(text):
    """
    检测极端的重复情况
    """
    if len(text) < 10:
        return False
    
    # 检测单个词的极端重复（如 "you you you you you"）
    words = text.lower().split()
    if len(words) >= 5:  # 至少5个词
        word_counts = {}
        for word in words:
            if len(word) <= 4:  # 只检测短词的重复，避免误杀长词
                word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            max_count = max(word_counts.values())
            # 更严格的阈值：90%以上才判定
            if len(words) >= 10 and max_count / len(words) > 0.9:
                return True
            # 或者：同一个短词连续出现5次以上
            if max_count >= 5:
                max_word = max(word_counts, key=word_counts.get)
                # 检查是否连续出现
                word_sequence = ' '.join(words)
                repeated_pattern = f"{max_word} " * 5
                if repeated_pattern.strip() in word_sequence:
                    return True
    
    # 检测单字符的极端重复（如 "哈哈哈哈哈哈哈哈哈哈"）
    clean_text = re.sub(r'[。，、！？!?\s,.!?]', '', text)
    if len(clean_text) >= 8:
        char_counts = {}
        for char in clean_text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        if char_counts:
            max_char_count = max(char_counts.values())
            # 某个字符连续出现8次以上
            if max_char_count >= 8:
                max_char = max(char_counts, key=char_counts.get)
                if max_char * 8 in clean_text:
                    return True
    return False


def has_obvious_gibberish(text):
    """
    检测明显的乱码
    """
    clean_text = re.sub(r'[。，、！？!?\s,.!?]', '', text)
    if len(clean_text) < 10:
        return True
    
    # 只检测是否几乎没有正常字符（汉字、字母、数字）
    normal_chars = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', clean_text)
    # 超过80%都不是正常字符，才判定为乱码
    if len(normal_chars) / len(clean_text) < 0.2:
        return True
    
    return False


def merge_results(segments_results, do_search=False):
    """合并所有分段结果"""
    if not segments_results:
        return {
            "query": "",
            "summary": "",
            "search": "",
            "elements": [],
            "del_flag": '1',
            "segments": []
        }
    
    all_texts = []
    seen_fingerprints = set()
    
    INVALID_ASR_PATTERNS = ["文本；正文", "文本正文", "文本:正文", "文本：正文"]
    
    def is_invalid_asr(text):
        if not text:
            return True
        text_clean = text.strip().replace(" ", "").replace("；", "").replace(":", "").replace("：", "")
        return text_clean in ["文本正文", ""] or text.strip() in INVALID_ASR_PATTERNS
    
    has_valid_asr = False
    has_valid_ocr = False
    
    for seg in segments_results:
        if not seg or not isinstance(seg, dict):
            continue
        
        text_dicts = seg.get("text_dicts", {})
        if not text_dicts:
            continue
        
        asr_text = text_dicts.get("audio_text_zh", "")
        if asr_text:
            asr_text = str(asr_text).strip()
        
        if asr_text and not is_invalid_asr(asr_text):
            has_valid_asr = True
            if all_texts:
                overlap_len = find_overlap(all_texts[-1], asr_text)
                if overlap_len > 0:
                    asr_text = asr_text[overlap_len:]
            
            if asr_text:
                all_texts.append(asr_text)
        
        ocr_text = text_dicts.get("video_text_zh", "")
        if ocr_text:
            ocr_text = str(ocr_text).strip()
        
        if ocr_text:
            ocr_sentences = extract_useful_ocr(ocr_text)
            
            if ocr_sentences:
                has_valid_ocr = True
            
            for sent in ocr_sentences:
                fingerprint = sent[:30]
                
                if fingerprint in seen_fingerprints:
                    continue
                
                combined_text = "".join(all_texts)
                if is_substring_in_text(sent, combined_text):
                    continue
                
                all_texts.append(sent)
                seen_fingerprints.add(fingerprint)
    
    full_query = "".join(all_texts) if all_texts else ""
    
    if full_query:
        full_query = re.sub(r'\s+', '', full_query)
        full_query = re.sub(r'。+', '。', full_query)
        full_query = full_query.strip('。，、 ')
        
        if full_query and full_query[-1] not in ['。', '!', '?']:
            full_query += '。'
    
    # ========== 提前判断del_flag ==========
    del_flag = '0'
    no_valid_sources = not has_valid_asr and not has_valid_ocr
    text_too_short = len(full_query) < 20
    only_punctuation = len(re.sub(r'[。，、！？!?\s]', '', full_query)) < 10
    is_extreme_repetition = has_extreme_repetition(full_query)
    is_obvious_gibberish = has_obvious_gibberish(full_query)

    if no_valid_sources:
        del_flag = '1'
        print("警告: 视频没有有效的ASR和OCR文本，标记为删除")
    elif text_too_short:
        del_flag = '1'
        print(f"警告: 合并文本过短(仅{len(full_query)}字符)，标记为删除")
        print(f"文本内容: {full_query}")
    elif only_punctuation:
        del_flag = '1'
        print(f"警告: 文本几乎全是标点符号，标记为删除")
        print(f"文本内容: {full_query}")
    elif is_extreme_repetition:
        del_flag = '1'
        print(f"警告: 文本包含极端重复内容，标记为删除")
        print(f"文本内容: {full_query}")
    elif is_obvious_gibberish:
        del_flag = '1'
        print(f"警告: 文本包含明显乱码，标记为删除")
        print(f"文本内容: {full_query}")


    # ========== 如果已标记删除,直接返回,不做后续处理 ==========
    if del_flag == '1':
        return {
            "query": full_query,
            "summary": "",
            "search": "",
            "elements": [],
            "del_flag": del_flag,
            "segments": segments_results
        }
    
    # ========== 只有del_flag=0时才继续处理 ==========
    all_element_dicts = []
    
    for seg in segments_results:
        if not seg or not isinstance(seg, dict):
            continue
        
        segment_id = seg.get("segment_id")
        elem_dicts = seg.get("element_dicts", [])
        if not elem_dicts:
            continue
        
        for elem_dict in elem_dicts:
            if not elem_dict or not isinstance(elem_dict, dict):
                continue
            
            elements = elem_dict.get("elements", [])
            if not elements:
                continue
            
            img_path = elem_dict.get("img", "")
            frame_id = None
            
            if img_path:
                parts = img_path.split('/')
                if len(parts) >= 2:
                    try:
                        frame_id = int(parts[-2])
                    except (ValueError, IndexError):
                        frame_id = None
            
            for elem_name in elements:
                elem_data = {
                    "segment_id": segment_id,
                    "frame_id": frame_id,
                    "elements": [elem_name],
                    "class": elem_dict.get("class"),
                    "local_timestamp": elem_dict.get("local_timestamp"),
                    "global_timestamp": elem_dict.get("global_timestamp", 0),
                }
                
                all_element_dicts.append(elem_data)
    
    all_element_dicts = sorted(
        all_element_dicts,
        key=lambda x: x.get("global_timestamp", 0)
    )
    
    summary = ""
    if full_query:
        try:
            summary = call_qwen_smart(full_query, all_element_dicts)
            print("生成摘要完成")
        except Exception as e:
            print(f"生成摘要时出错: {e}")
            summary = ""
    else:
        if all_element_dicts:
            element_names = []
            seen_names = set()
            for elem in all_element_dicts:
                name = elem.get("elements", [""])[0]
                if name and name not in seen_names:
                    element_names.append(name)
                    seen_names.add(name)
                    if len(element_names) >= 10:
                        break
            
            element_summary = "、".join(element_names)
            summary = f"视频包含以下视觉元素：{element_summary}"
        else:
            summary = "该视频无可识别的文本或视觉元素内容"
            print("警告: 视频没有有效的文本内容和视觉元素")
    
    search_result = {}
    if do_search is True:
        search_result = query_search(query_sentence=full_query)
    
    return {
        "query": full_query,
        "summary": summary,
        "elements": all_element_dicts,
        "search": search_result,
        "del_flag": del_flag,
        "segments": segments_results
    }


def fl_long_main(
    input_video,
    work_dir="fl_tmp_seg",
    audio_model=None,
    audio_processor=None,
    sam=None,
    class_model=None,
    processor=None,
    CLIP_MODEL=None,
    ocr_fl=None,
    sentence_model=None,
    ner_pipeline=None,
    do_search=True,
    progress_callback=None
):
    """长视频处理主函数"""
    def update_progress(progress, message=''):
        if progress_callback:
            try:
                progress_callback(min(progress, 99), message)
            except Exception as e:
                print(f"进度回调失败: {e}")
    
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"视频文件不存在: {input_video}")
    
    update_progress(1, '正在读取视频信息...')
    
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_video
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0 or not result.stdout.strip():
        raise Exception(f"无法获取视频时长: {result.stderr}")
    
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        raise Exception(f"视频时长格式错误: {result.stdout}")
    
    print(f"视频时长: {duration:.2f}秒")
    update_progress(5, f'视频时长 {duration:.1f}秒')
    
    if duration < 60:
        update_progress(10, '检测到短视频，开始处理...')
        print("检测到短视频，直接处理...")
        
        try:
            element_dicts, text_elements, text_dicts = fl_video_main(
                input_video,
                audio_model=audio_model,
                audio_processor=audio_processor,
                sam=sam,
                class_model=class_model,
                processor=processor,
                CLIP_MODEL=CLIP_MODEL,
                ocr_fl=ocr_fl,
                sentence_model=sentence_model,
                ner_pipeline=ner_pipeline,
                prefilled_audio_fl=None,
                prefilled_audio_zh=None,
                segment_start_time=0.0,
                progress_callback=progress_callback,
                progress_range=(10, 90)
            )
            update_progress(90, '视频分析完成')
        except Exception as e:
            print(f"短视频处理失败: {e}")
            raise
        
        update_progress(92, '正在生成结果...')
        seg_result = {
            "segment_id": 0,
            "element_dicts": element_dicts if element_dicts else [],
            "text_elements": text_elements if text_elements else {},
            "text_dicts": text_dicts if text_dicts else {}
        }
        
        final_result = merge_results([seg_result], do_search=do_search)
        update_progress(96, '正在保存结果...')
        
        out_json = os.path.splitext(input_video)[0] + "_analysis.json"
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            print(f"短视频处理完成，结果已保存到 {out_json}")
        except Exception as e:
            print(f"警告: 保存结果文件失败: {e}")
        
        update_progress(99, '短视频处理完成')
        return final_result
    
    full_audio = None
    try:
        update_progress(6, '正在准备工作目录...')
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=True)
        
        update_progress(8, '正在切片视频...')
        slice_video(input_video, work_dir, segment_seconds=60)
        
        seg_files = sorted(glob.glob(os.path.join(work_dir, "segment_*.mp4")))
        total_segments = len(seg_files)
        update_progress(15, f'已切分为 {total_segments} 个片段')
        
        update_progress(15, '正在提取音频并转录...')
        try:
            full_audio, full_asr_segments = extract_full_audio(
                input_video, 
                audio_model=audio_model, 
                audio_processor=audio_processor
            )
            update_progress(25, f'音频转录完成，识别到 {len(full_asr_segments)} 个片段')
        except Exception as e:
            print(f"警告: 音频提取失败: {e}")
            full_audio = None
            full_asr_segments = []
            update_progress(25, '音频提取失败，跳过音频处理')
        
        update_progress(25, f'开始处理 {total_segments} 个视频片段...')
        
        seg_results = process_segments(
            work_dir,
            full_asr_segments=full_asr_segments,
            audio_model=audio_model,
            audio_processor=audio_processor,
            sam=sam,
            class_model=class_model,
            processor=processor,
            CLIP_MODEL=CLIP_MODEL,
            ocr_fl=ocr_fl,
            sentence_model=sentence_model,
            ner_pipeline=ner_pipeline,
            progress_callback=update_progress
        )
        
        update_progress(90, '片段处理完成，正在合并结果...')
        
        update_progress(90, '正在合并分析结果...')
        final_result = merge_results(seg_results, do_search=do_search)
        update_progress(95, '结果合并完成')
        
        update_progress(95, '正在保存结果文件...')
        out_json = os.path.splitext(input_video)[0] + "_analysis.json"
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            print(f"长视频处理完成，结果已保存到 {out_json}")
            update_progress(99, '长视频处理完成')
        except Exception as e:
            print(f"警告: 保存结果文件失败: {e}")
        
        return final_result
    
    finally:
        if full_audio and os.path.exists(full_audio):
            try:
                os.remove(full_audio)
            except:
                pass