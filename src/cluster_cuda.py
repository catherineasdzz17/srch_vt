import cv2
import os
import shutil
import torch

import numpy as np
import torch.nn as nn
# import matplotlib.pyplot as plt

from transformers import ViTImageProcessor, ViTModel
from frame_utils import frame_filter, get_features
from config import CLASS_MODEL_PATH


def distance(x, y):
    '''
    计算两向量的欧式距离
    '''
    return np.linalg.norm(x - y)


def getDistanceMatrix(data):
    '''
    计算样本间的距离矩阵
    '''
    N, D = data.shape  # 样本数量，样本维度
    dists = torch.zeros([N, N])
    # 用于计算两个张量间的欧氏距离
    pdist = nn.PairwiseDistance(p=2, eps=0)
    # 遍历每个样本
    for i in range(N):
        tmp = data[i, :].unsqueeze(0)
        tmp = tmp.repeat((N, 1))
        dists[i] = pdist(tmp, data)  # 计算与其他样本的距离
    return dists


def select_dc(dists):
    '''
    计算距离矩阵的截断距离dc
    '''
    # 不使用np.percentile()，因为dists有部分重复值、0值
    N, D = dists.shape  # 样本数量，样本维度
    dists_flat = torch.reshape(dists, (1, N * N)).squeeze(0)  # 展平矩阵
    dists_sort, dists_id = torch.sort(dists_flat)  # 从小到大排序
    position = int(N * (N - 1) * 0.002)  
    # 将距离矩阵排序后取第2%个，可以调整该百分比使聚类效果更加
    dc = dists_sort[position + N]  # 跳过对角线为0的元素
    return dc.item()


def calculate_rho(dists, dc, method='truncated'):
    """
    计算每个点的局部密度ρ(rho)
    选择方法默认为截断核；传入任意参数，即可选择高斯核
        建议小规模数据选择高斯核，因为该方法计算出来带小数，比较更准确
        大规模数据选择截断核，效率更高
    """
    N = dists.shape[0]
    rho = torch.zeros(N)  # 初始化密度向量
    # 记录与每个点与dc的大小关系
    tmp = torch.full([N, N], dc)
    bools = dists < tmp  
    # 截断核：适合大规模数据
    # 统计在dc区域内有多少个点
    if method == 'truncated':
        tmp = torch.full([N, N], dc)
        bools = dists < tmp
        rho = torch.sum(bools, dim=1)
    # 高斯核：适合小规模数据
    else:
        # 遍历每个点与所有点的距离，按高斯核公式作为权重计算密度
        for i in range(N):
            rho[i] = torch.sum(torch.exp(-(dists[i, :] / dc) ** 2)).item()
    return rho


def calculate_deltas(dists, rho):
    '''
    计算每个点与密度比它高的点的最小距离δ
    '''
    N = dists.shape[0]
    deltas = torch.zeros(N)
    neighbors = torch.zeros(N)  # 密度更高，且距离最近的点的索引
    # 遍历每一个点
    for i in range(N):
        # 判定是否小于等于rho[i]的矩阵
        tmp = rho <= rho[i].item()
        # 统计小于等于的个数
        num_zero = tmp.sum().item()
        if num_zero == N:  # 当前点密度最大
            deltas[i] = torch.max(dists[i]).item()
            continue
        # 统计大于的距离（小于等于的全部变成0了）
        tmp = (~tmp) * dists[i]
        # 排序，1为结果，2为id
        tmp_sort, tmp_id = torch.sort(tmp)
        deltas[i] = dists[i][tmp_id[num_zero]]
        neighbors[i] = tmp_id[num_zero]
    return deltas, neighbors


def clusters(dists, rho, deltas, neighbors, k=0):
    """
    选择聚类中心并进行聚类
    k默认为0，代表自动确定中心点，若传入值则确定多少个中心点
    """
    N, D = dists.shape  # 样本数量和距离维度
    labels = torch.full((1, N), -1).squeeze(dim=0)  # 初始化所有label为-1，表示未分配类别

    # 自动确定聚类中心
    if k == 0:
        # 选择ρ×δ最大的前10%作为聚类中心（高密度且距离其他高密度较远）
        rho_deltas = rho * deltas
        rho_deltas_sort, rho_deltas_id = torch.sort(rho_deltas, descending=True)
        rho_deltas_sort = rho_deltas_sort[0 : int(N * 0.1)]
        rho_deltas_id = rho_deltas_id[0 : int(N * 0.1)]
        centers = rho_deltas_id
        for i, center_id in enumerate(centers):
            labels[center_id.item()] = i

    else:
        # 手动输入聚类数k,选择ρ×δ最大的k个作为中心点，并聚类
        rho_deltas = rho * deltas
        centers = torch.argsort(-rho_deltas)
        for i in range(k):
            labels[centers[i].item()] = i
            # 画中心点
            # plt.scatter(data[center[i], 0], data[center[i], 1], color="k", marker="+", s=250)

    # 对非中心点进行聚类
    # 从密度最高开始，将他们分给比它们密度更高且距离最近的点的类
    rho_sort = torch.argsort(-rho)
    for i in range(0, N):
        if labels[rho_sort[i].item()] == -1:
            labels[rho_sort[i].item()] = labels[int(neighbors[rho_sort[i].item()].item())]
    return labels, centers[0 : k]


def DPC(data, k):
    '''
    DPC聚类算法
    '''
    dists = getDistanceMatrix(data)  # 计算距离矩阵
    dc = select_dc(dists)  # 选取截断距离dc
    rho = calculate_rho(dists, dc, 'gauss')  # 计算局部密度rho
    deltas, neighbors = calculate_deltas(dists, rho)  # 计算delta
    # 归一化
    rho = rho / torch.sum(rho)
    deltas = deltas ** 2
    labels, centers = clusters(dists, rho, deltas, neighbors, k)  # 聚类

    # # 画出rho与deltas
    # print('show_imging')
    # for i in range(data.shape[0]):
    #    plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
    #    plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
    #    plt.xlabel("rho")
    #    plt.ylabel("deltas")
    # plt.show()
    return labels, centers


def dpc_main(source, output_path='', isfile=False, dpc_rate=0.3):
    '''
    DPC聚类的主函数，抽取视频关键帧
    输入：视频逐帧rgb矩阵的列表或者包含所有图片的文件夹
    输出：将关键帧输出到指定文件夹的路径中去（默认），或者关键帧的rgb矩阵列表
    '''
    # 准备图像特征提取模型
    processor = ViTImageProcessor.from_pretrained(CLASS_MODEL_PATH)
    model = ViTModel.from_pretrained(CLASS_MODEL_PATH)
    features = []

    print('Extracting features------------------')
    # 从文件夹加载图像
    if isinstance(source, str):
        source_path = source
        source = []
        # 逐文件加载并转为RGB格式
        for i in os.listdir(source_path):
            path = os.path.join(source_path, i)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            source.append(img)
    # 提取特征向量
    with torch.no_grad():
        for img in source:
            feature = get_features(img, processor, model)
            features.append(feature)
    features = torch.cat(features)  # 拼接为特征矩阵

    print('DPC--------------------------')
    # DPC聚类，最多聚3类
    # labels, centers = DPC(features, int(min(3, features.shape[0] * dpc_rate)))
    labels, centers = DPC(features, int(min(5, features.shape[0] * dpc_rate)))
    print('Shiltering--------------------')
    # 去除相似帧
    centers = frame_filter(source, centers)
    centers = [i.item() for i in centers]
    # 保存关键帧
    if isfile:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        num = 0
        for j in centers:
            cv2.imwrite(os.path.join(output_path, f'{num}.jpg'), source[j])
            num += 1
    print(centers)
    # 返回centers
    return centers