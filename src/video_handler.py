import cv2
import logging
import os
import random
import shutil
import time
import torch

import Levenshtein
import zhconv

import numpy as np
import pandas as pd

from modelscope import pipeline, Tasks
from paddleocr import PaddleOCR
from tqdm import tqdm
from transformers import ViTImageProcessor
from FlagEmbedding import BGEM3FlagModel

from cluster_cuda import dpc_main
from encode import ChineseCLIP
from frame_utils import check_repetition
from img_helpers import img_recg
from my_utils import element_filter
from segment_helpers import Segment_generator, get_segment_images, get_imgs
from vit_model import vit_model
from config import CLASS_MODEL_PATH, CLASS_CHECKPOINT


def read_mov(source):
    '''
    生成视频逐帧输出的迭代器
    输入：视频路径或者已经构建好的videocapture对象
    '''
    # 根据路径读取视频
    if isinstance(source, str):
        source = cv2.VideoCapture(source)
    fps = source.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率fps

    # 读取视频帧
    while True:
        # 读取一帧，ret是布尔值，表示是否读取成功，frame是该帧的图像内容
        ret, frame = source.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield ret, frame, int(fps)
        else:
            break
    # 读取结束
    yield False, None, -1


def video_handler_main(
        source,
        ids=None,
        audio_txts=None,
        sam=None,
        class_model=None,
        processor=None,
        CLIP_MODEL=None,
        ocr_ch=None,
        ocr_chinese_cht=None,
        sentence_model=None,
        ner_pipeline=None,
        seg_pic_save_path=None
):
    '''
    操作视频的主函数
    输入为路径或videocapture类(source)及代号(ids)，以及音频转换的文本(audio_txts)
    输出包含两部分：
        1.视频中出现的关键要素
        2.视频的文字信息内容
    '''
    # 加载视频迭代器
    g = read_mov(source)

    # 初始化
    frames = []
    frame_txts_ch = [[]]  # 简体文字帧文本列表
    frame_txts_chinese_cht = [[]]  # 繁体文字帧文本列表
    frame_persons = []  # 检测到的人物
    frame_nations = []  # 检测到的国家
    frame_others = []  # 检测到的其他类别元素
    element_dicts = []  # 所有识别到的元素的结构化信息
    key_words = []  # 提取的关键词
    num = -1  # 当前帧编号

    a = time.time()
    while True:
        # 逐帧处理，带下采样，一秒选一帧
        flag, frame, fps = next(g)  # 获取下一帧及其帧率
        num += 1
        if not flag:  # 视频结束
            break
        if num % fps != 3:  # 采样，每秒一帧
            continue
        frames.append(frame)

        try:
            # 简体中文OCR
            results_ch = ocr_ch.ocr(frame, cls=True)[0]
            tmp_results_ch = []
            for result in results_ch:
                prob = result[1][1]
                if prob >= 0.9:
                    tmp_results_ch.append(result[1][0])
            # 对OCR结果去重（去除与上一帧内容高度重复的结果）
            if len(tmp_results_ch) == 0 or check_repetition(frame_txts_ch[-1], tmp_results_ch):
                continue
            frame_txts_ch.append((tmp_results_ch, num))

            # 繁体中文OCR
            results_chinese_cht = ocr_chinese_cht.ocr(frame, cls=True)[0]
            tmp_results_chinese_cht = []
            for result in results_chinese_cht:
                prob = result[1][1]
                if prob >= 0.9:
                    tmp_results_chinese_cht.append(result[1][0])
            # 对OCR结果去重
            if len(tmp_results_chinese_cht) == 0 or check_repetition(frame_txts_chinese_cht[-1],
                                                                     tmp_results_chinese_cht):
                continue
            frame_txts_chinese_cht.append((tmp_results_chinese_cht, num))
        except:
            continue
    b = time.time()
    print(f'ocr: {b-a}')

    # dpc聚类抽取视频关键帧，数字为list中的帧ID
    key_frames_ids = dpc_main(frames, output_path='', isfile=False)
    # 元组包含内容：帧矩阵与帧ID
    key_frames = [(frames[i], i) for i in key_frames_ids]
    # 保存上面选出来的关键帧的BGR矩阵
    if os.path.exists(f'tmp/key_frames'):
        shutil.rmtree(f'tmp/key_frames')
    os.mkdir(f'tmp/key_frames')
    for key_frame in key_frames:
        cv2.imwrite(f'tmp/key_frames/{key_frame[1]}.jpg', key_frame[0])
    
    c = time.time()
    print(f'dpc: {c-b}')

    # 关键帧切分与元素识别
    if os.path.exists(f'tmp/segment_frames'):
        shutil.rmtree(f'tmp/segment_frames')
    os.mkdir(f'tmp/segment_frames')

    for key_frame, key_frames_id in key_frames:
        # 将图片切分完后保存在指定位置
        os.mkdir(f'tmp/segment_frames/{key_frames_id}')
        # SAM切分图像
        get_segment_images(sam, key_frame, f'tmp/segment_frames/{key_frames_id}')
        # 获取切割后的子图
        seg_pics, fs = get_imgs(f'tmp/segment_frames/{key_frames_id}')
        i = 0

        for seg_pic in seg_pics:
            # 识别切割后的每张子图
            element_dict, keyword = img_recg(processor, class_model, CLIP_MODEL, seg_pic)
            if element_dict['class'] == 0:
                frame_persons.append(element_dict['elements'])
            elif element_dict['class'] == 1:
                frame_nations.append(element_dict['elements'])
            elif element_dict['class'] == 2:
                frame_others.append(element_dict['elements'])
            if element_dict['class'] != 3:
                element_dicts.append(element_dict)
                key_words.append(keyword)

    d = time.time()
    print(f'recg: {d-c}')

    # 合并文字结果并去重简化
    ans_lst = frame_txts_ch[1][0]
    for i in range(2, len(frame_txts_ch)):
        tmp_lst = frame_txts_ch[i][0]
        for eve in tmp_lst:
            if eve not in ans_lst:
                ans_lst.append(eve)
    ans_str = '，'.join(ans_lst)
    ans_str = zhconv.convert(ans_str, 'zh-hans')[0:510]
    element_dicts = element_filter(element_dicts)  # 过滤无效元素

    print(element_dicts)
    return element_dicts, ans_str


def chinese_ner(sentence, ner_pipeline):
    '''
    命名实体识别
    '''
    # 命名实体识别
    result = ner_pipeline(sentence)
    output_lst = result['output']
    # 结构化输出
    ans = {'PER': [], 'LOC': [], 'ORG': []}
    for entity in output_lst:
        ans[entity['type']].append(entity['span'])
    return ans


def ocr_list_handler(ocr_l, audio_txts):
    '''
    匹配处理ocr文本和音频文本
    '''
    can_l = []   # 记录已处理过的文本
    fin_l = []   # 最终的输出结果列表

    # 遍历OCR结果
    for i in range(1, len(ocr_l)):
        tmp_l = ocr_l[i][0]  # 提取该帧的OCR文本列表
        flag = True
        for tmp in tmp_l:
            if len(tmp) < 6:  # 跳过过短的文本
                continue
            for can in can_l:
                if Levenshtein.jaro(tmp, can) > 0.8:  # 处理重复文本
                    flag = False
                    break
            if flag:  # 新文本
                simple_tmp = zhconv.convert(tmp, 'zh-cn')  # 转为简体
                maxs = ''  # 最相似文本
                maxj = 0  # 最大相似度
                # 在音频文本中查找最匹配文本
                for j in audio_txts:
                    j = j.strip()
                    if Levenshtein.jaro(j, simple_tmp) > maxj:
                        maxj = Levenshtein.jaro(j, simple_tmp)
                        maxs = j
                # 原始OCR文本，简体文本，最匹配音频文本，帧编号
                fin_l.append((tmp, simple_tmp, maxs, ocr_l[i][1]))
                can_l.append(tmp)
    return fin_l


def video_tag_main(ids):
    '''
    视频标注主函数，处理批量视频数据至csv格式
    '''
    with torch.no_grad():
        df = pd.read_csv(f'video_csv/part_{ids}.csv')
        rows_to_process = df[df['tag_exist'] == False]  # 未处理视频数据

        # 遍历未处理的视频并分析
        for index, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process)):
            video_path = row['video_path']
            try:
                ans = video_handler_main(video_path, ids=ids, sam=SAM, processor=processor, class_model=class_model,
                                         CLIP_MODEL=CLIP_MODEL, ocr_chinese_cht=ocr_chinese_cht, ocr_ch=ocr_ch)
                df.at[index, 'frame_txts_ch'] = str(ans[1])
                df.at[index, 'frame_txts_chinese_cht'] = str(ans[2])
                df.at[index, 'frame_persons'] = " | ".join(ans[3])
                df.at[index, 'frame_nations'] = " | ".join(ans[4])
                df.at[index, 'frame_others'] = " | ".join(ans[5])
                df.at[index, 'tag_exist'] = True
                df.to_csv(f'video_csv/part_{ids}.csv', index=False)
            except:
                continue


def video_data_gen():
    '''
    视频数据准备
    '''
    # 遍历所有视频文件
    video_dir = 'videos'
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    # 生成初始字段
    random_numbers = [random.randint(1, 20) for _ in range(len(video_paths))]
    tag_exist = [False for _ in range(len(video_paths))]
    tmp = [None for _ in range(len(video_paths))]

    # 初始化DataFrame
    df = pd.DataFrame({
        'video_path': video_paths,
        'random_number': random_numbers,
        'tag_exist': tag_exist,
        'ID': None,
        'frame_txts_ch': None,
        'frame_txts_chinese_cht': None,
        'frame_persons': None,
        'frame_nations': None,
        'frame_others': None
    })
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    # 计算每部分的大小
    num_parts = 6
    part_size = len(df_shuffled) // num_parts

    # 分割DataFrame并保存为新的CSV文件
    for i in range(num_parts):
        start_idx = i * part_size
        end_idx = (i + 1) * part_size if i < num_parts - 1 else None
        part_df = df_shuffled.iloc[start_idx:end_idx]
        part_file_path = f'video_csv/part_{i + 1}.csv'
        part_df.to_csv(part_file_path, index=False)
        print(f"已保存 {part_file_path}")

    # 处理最后一部分，确保所有数据都被包含
    if end_idx is not None and end_idx < len(df_shuffled):
        part_df = df_shuffled.iloc[end_idx:]
        part_file_path = f'video_csv/part_{num_parts}.csv'
        part_df.to_csv(part_file_path, index=False)
        print(f"已保存 {part_file_path}")


def find_top_five_similar(input_embed):
    '''
    从预定的文本向量中查找与输入向量最相似的5个内容
    '''
    candidate_lst = torch.load('data/video_txt.pkl')  # 候选数据
    # 计算每个元组中嵌入向量与输入嵌入向量的余弦相似度
    similarities = []
    for path, text, embed in candidate_lst:
        dot_product = np.dot(embed, input_embed)
        norm_embed = np.linalg.norm(embed)
        norm_input = np.linalg.norm(input_embed)
        similarity = dot_product / (norm_embed * norm_input)
        similarities.append((text, similarity))

    # 根据相似度降序排序
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    # 取前五个
    top_five = sorted_similarities[:5]
    return top_five


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    # 分类模型，分类图片是人/旗帜
    processor = ViTImageProcessor.from_pretrained(CLASS_MODEL_PATH)
    class_model = vit_model().to('cuda:0')
    class_model.load_state_dict(torch.load(CLASS_CHECKPOINT))

    CLIP_MODEL = ChineseCLIP()
    # 切图
    SAM = Segment_generator()
    SAM = SAM.get_model()

    ner_pipeline = pipeline(Tasks.named_entity_recognition, model='model/bert_ner', device='cuda:0')
    # 简体：ch，繁体：chinese_cht，showlog：是否输出日志
    ocr_chinese_cht = PaddleOCR(use_angle_cls=True, lang="chinese_cht", show_log=False, device='cuda:0')
    ocr_ch = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, device='cuda:0')
    sentence_model = BGEM3FlagModel('model/bge-m3', use_fp16=True, device='cuda:0')

    # 输入视频路径
    video_path = '/home2/dxy/srch_vt/final_test/video_example_data/Q0xzZZuhlx080P0BvGcw01041200iS2F0E010.mp4'

    element_dicts, query_sentence = video_handler_main(video_path, sam=SAM, processor=processor, class_model=class_model,
                             CLIP_MODEL=CLIP_MODEL, ocr_chinese_cht=ocr_chinese_cht, ocr_ch=ocr_ch,
                             sentence_model=sentence_model, ner_pipeline=ner_pipeline)
    print(element_dicts, query_sentence)