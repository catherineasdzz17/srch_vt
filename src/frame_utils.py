import cv2
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from segment_helpers import get_segment_images_v2


def cal_similarity(raw_pics, i, j):
    '''
    计算两张图像的像素级相似度
    '''
    i = raw_pics[i]
    j = raw_pics[j]
    return np.count_nonzero(i == j) / i.size


def frame_filter(raw_pics, centers, flag=0.8):
    '''
    检查DPC聚类出来的中心帧之间的相似度，如果大于指定值就舍弃该帧，默认0.8
    '''
    ans = [centers[0]]  # 初始化保留帧列表
    # 遍历所有中心帧
    for i in centers:
        tmp_flag = True
        for j in ans:  # 和已保留帧比较
            # 相似度过大则舍弃
            if cal_similarity(raw_pics, i, j) > flag:
                tmp_flag = False
                break
        if tmp_flag:
            ans.append(i)
    return ans


def get_features(source, processor=None, model=None):
    '''
    提取图片特征
    输入：图片路径或者顺序为RGB的ndarray矩阵
    输出：1*768的特征向量
    '''
    if processor is None:
        # 加载预训练的ViT处理器
        processor = ViTImageProcessor.from_pretrained('model/ViT_extractor')
    if model is None:
        # 加载预训练的ViT模型
        model = ViTModel.from_pretrained('model/ViT_extractor')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # 输入为路径，读取图像并转为RGB
    if isinstance(source, str):
        source = cv2.imread(source)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        # 转换为可输入格式
        inputs = processor(images=source, return_tensors="pt")
        inputs = inputs.to(device)
        # 前向传播提取特征
        outputs = model(**inputs).last_hidden_state[:, 0, :]
    return outputs


def check_repetition(a, b):
    '''
    判断ocr识别出来的文字内容是否相同
    输入：两帧OCR识别的列表
    输出：True or False
    '''
    # 文本长度不同，返回False
    if len(a) != len(b):
        return False
    # 遍历文本a
    for i in range(0, len(a)):
        str_a = a[i]
        str_b = b[i]
        # 对应行长度不同，返回False
        if len(str_a) != len(str_b):
            return False
        total_l = len(str_a)
        num = 0
        # 每个字符逐一比较
        for j in range(0, total_l):
            if str_a[j] != str_b[j]:
                num += 1
        # 不一致比例超过20%，认为不同，返回False
        if num / total_l > 0.2:
            return False
    return True


def frame_recg(frame, sam, class_model):
    '''
    识别帧中的国旗，人物，武器装备等
    输入：图片的rgb矩阵
    输出：包含国旗、人物、武器装备的列表（没有就返回空列表）
    '''
    # 使用切割工具获得切割后需要分类/识别的帧
    seg_pics = get_segment_images_v2(sam, frame)