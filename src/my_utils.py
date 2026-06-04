import collections
import json
import math
import os
import re
import requests
import torch
import zhconv
import numpy as np
from cn_clip.clip import load_from_name
from PIL import Image
from typing import Dict, Any, Optional


def find_elements(input_str, candidate_set):
    '''
    查找元素对应的值
    '''
    for eve in candidate_set.keys():
        if input_str in eve:
            return candidate_set[eve]
    return None


def trans_ocr(list1, list2):
    '''
    将繁体简体的ocr结果配对处理，给大模型判断正误
    '''
    max_length = max(len(list1), len(list2))
    # 若长度不同，用特殊无意义字符填充对齐
    padded_list1 = list1 + ['donvwoafncwCDASCAC ACASCQAVQACA vw'] * (max_length - len(list1))
    padded_list2 = list2 + ['donvwoafCZXCXZVDAWV ADCASQSCQASncw vw'] * (max_length - len(list2))
    # 将对齐的文本均转换为简体
    return [(zhconv.convert(padded_list1[i], 'zh-hans'), zhconv.convert(padded_list2[i], 'zh-hans')) for i in range(max_length)]


def call_qwen(data, ctype='generate_elements'):
    '''
    QWen2.5接口调用
    '''
    base_url = 'http://127.0.0.1:14700'   
    if ctype == 'generate_elements':
        url = f'{base_url}/generate_elements'
        payload = {'text': data}
    elif ctype == 'judge_sim_tra':
        url = f'{base_url}/judge_sim_tra'
        payload = {'sim': data[0], 'tra': data[1]}
    elif ctype == 'add_punctuation':
        url = f'{base_url}/add_punctuation'
        payload = {'text': data}
    elif ctype == 'judge_similarity':
        url = f'{base_url}/judge_similarity'
        payload = {'a': data[0], 'b': data[1]}
    elif ctype == 'generate_yinyu_fengci':
        url = f'{base_url}/generate_yinyu_fengci'
        payload = {'text': data}
    elif ctype == 'generate_summary':
        url = f'{base_url}/generate_summary'
        payload = {'text': data[0], 'element': data[1]}
    
    elif ctype == 'generate_summary_chunk':
        url = f'{base_url}/generate_summary_chunk'
        payload = {'text': data[0], 'element': data[1]}
    
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    result = response.json()
    
    # 统一拆包：所有 API 都返回字符串
    if ctype in ['generate_summary', 'generate_summary_chunk']:
        return result.get('summary', '')
    else:
        return result.get('result', '')


def element_filter(element_dicts):
    '''
    过滤重复或无效元素
    '''
    # 国家及无效词
    # element_lst = ['Unknown','None','利比亚','中国','杂物','圣卢西亚','马绍尔群岛','贝宁','马来西亚',"利比果亚","西萨摩亚","比利时","安哥拉",
    # "阿富汗", "阿尔巴尼亚", "阿尔及利亚", "安道尔", "安哥拉", "安提瓜和巴布达","爱沙尼亚",
    # "阿根廷", "亚美尼亚", "阿塞拜疆", "巴林", "孟加拉国", "巴巴多斯", "白俄罗斯", 
    # "伯利兹", "贝宁", "不丹", "玻利维亚", "波斯尼亚和黑塞哥维那", "博茨瓦纳", 
    # "巴西", "文莱", "保加利亚", "布基纳法索", "布隆迪", "柬埔寨", "喀麦隆", 
    # "佛得角", "中非共和国", "乍得", "智利", "中国", "哥伦比亚", "科摩罗", 
    # "刚果民主共和国", "刚果共和国", "科特迪瓦", "克罗地亚", "古巴", "塞浦路斯", 
    # "吉布提", "多米尼加", "多米尼克", "厄瓜多尔", "埃及", "萨尔瓦多", "赤道几内亚", 
    # "加蓬", "冈比亚", "格鲁吉亚", "加纳", "格林纳达", "危地马拉", "几内亚", 
    # "几内亚比绍", "圭亚那", "海地", "洪都拉斯", "印度", "印度尼西亚", "伊朗", 
    # "伊拉克", "牙买加", "约旦", "哈萨克斯坦", "肯尼亚", "基里巴斯", "科威特", 
    # "吉尔吉斯斯坦", "老挝", "黎巴嫩", "莱索托", "利比里亚", "利比亚", "列支敦士登", 
    # "马达加斯加", "马拉维", "马来西亚", "马尔代夫", "马里", "马耳他", "马绍尔群岛", 
    # "毛里塔尼亚", "毛里求斯", "墨西哥", "摩尔多瓦", "摩纳哥", "蒙古", "黑山", 
    # "蒙特内哥罗", "摩洛哥", "莫桑比克", "缅甸", "纳米比亚", "瑙鲁", "尼泊尔", 
    # "尼加拉瓜", "尼日尔", "尼日利亚", "北马其顿", "阿曼", "巴基斯坦", "帕劳", 
    # "巴勒斯坦", "巴拿马", "巴布亚新几内亚", "巴拉圭", "秘鲁", "菲律宾", "卡塔尔", 
    # "罗马尼亚", "俄罗斯", "卢旺达", "圣基茨和尼维斯", "圣卢西亚", "圣文森特和格林纳丁斯", 
    # "圣马力诺", "Sao Tome and Principe", "沙特阿拉伯", "塞内加尔", "塞尔维亚", "塞舌尔", 
    # "塞拉利昂", "所罗门群岛", "索马里", "南非", "南苏丹", "斯里兰卡", "苏丹", 
    # "苏里南", "叙利亚", "塔吉克斯坦", "坦桑尼亚", "泰国", "东帝汶", "多哥", 
    # "汤加", "特立尼达和多巴哥", "突尼斯", "土耳其", "土库曼斯坦", "图瓦卢", 
    # "乌干达", "乌克兰", "阿联酋", "乌拉圭", "乌兹别克斯坦", "瓦努阿图", "梵蒂冈", 
    # "委内瑞拉", "越南", "也门", "赞比亚", "津巴布韦"]

    element_lst = ['Unknown','None', '杂物', '武器装备']
    
    new_dicts = []  # 有效元素
    # 元素过滤
    for element_dict in element_dicts:
        if element_dict['elements'][0] not in element_lst:
            new_dicts.append(element_dict)
            element_lst.append(element_dict['elements'][0])
    return new_dicts


def load_class_model(path='model/cn-clip/epoch200.pt'):
    '''
    加载分类模型
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name('ViT-B-16', device=device, download_root='model/cn-clip/')
    # 加载模型权重
    dic = torch.load(path)['state_dict']
    new_dic = collections.OrderedDict()
    for i in dic.keys():
        new_dic[i[7:100]] = dic[i]
    model.load_state_dict(new_dic) 
    model.eval()
    return model, preprocess, device


def encode_flags():
    '''
    编码旗帜特征向量
    '''
    path = 'flags'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name('ViT-B-16', device=device, download_root='model/cn-clip/')  # 加载模型
    dic = {}
    # 遍历目录下所有图像
    for i in os.listdir(path):
        img_path = os.path.join(path, i)
        img = Image.open(img_path)
        inputs = preprocess(img).unsqueeze(0).to(device)  # 转换为输入格式
        with torch.no_grad():
            # 图像编码
            img_tensor = model.encode_image(inputs)
            img_tensor /= img_tensor.norm(dim=-1, keepdim=True)
            img_tensor = img_tensor.squeeze(dim=0)
        dic[i.split('.')[0]] = img_tensor
    torch.save(dic, 'data/flag.pkl')  # 保存编码文件


def find_flags(path):
    '''
    旗帜图像相似度匹配
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name('ViT-B-16', device=device, download_root='model/cn-clip/')  # 加载模型
    img = Image.open(path)  # 加载输入图像
    inputs = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # 编码输入图像
        img_tensor = model.encode_image(inputs)  
        img_tensor /= img_tensor.norm(dim=-1, keepdim=True)
        img_tensor = img_tensor.squeeze(dim=0)
    dic = torch.load('data/glag_feature.pkl')  # 加载已知旗帜数据
    print(torch.sum(img_tensor * dic['韩国'], dim=-1))
    print(torch.sum(img_tensor * dic['中国'], dim=-1))
    # 计算相似度
    best_score = float('-inf')
    best_flag = 'Unknown'
    for name, vec in dic.items():
        similarity = torch.sum(img_tensor * vec).item()
        if similarity > best_score:
            best_score = similarity
            best_flag = name
    return best_flag


def rgba2rgb(img):
    '''
    将RGBA转换为RGB
    '''
    raw_array = np.asarray(img)  # 转为np数组
    out_array = np.ones((raw_array.shape[0], raw_array.shape[1], 3), dtype=np.uint8)
    # 遍历每个像素点
    for i in range(0, raw_array.shape[0]):
        for j in range(0, raw_array.shape[1]):
            # 检查该像素点的alpha值是否为0
            if raw_array[i][j][3] == 0:
                out_array[i][j] = [255, 255, 255]  # 透明像素RGB设置为白色
            else:
                out_array[i][j] = raw_array[i][j][0 : 3]  # 保留原始RGB值，忽略alpha
    out_img = Image.fromarray(out_array)  # 转换回图像格式
    return out_img


# def search_subimg(img, dtype, clip_model, clip_preprocessor):
#     '''
#     根据类型查找与输入图像最相似的图像
#     '''
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # clip_model, clip_preprocessor = load_from_name('ViT-B-16', device=device, download_root='model/cn-clip/')
#     inputs = clip_preprocessor(img).unsqueeze(0).to(device)
#     img_feature = clip_model.encode_image(inputs)  # 提取输入图像特征

#     candidate_feature = torch.load(f'data/{dtype}.pkl')  # 加载对应类型的特征向量数据

#     max_id = ''
#     max_sum = 0
#     img_feature /= img_feature.norm(dim=-1, keepdim=True)  # 归一化
#     img_feature = img_feature.squeeze(dim=0)
#     # 遍历所有特征向量
#     for i in candidate_feature.keys():
#         # 获得最相似图像
#         if torch.dot(img_feature, candidate_feature[i]).item() > max_sum:
#             max_id = i
#             max_sum = torch.dot(img_feature, candidate_feature[i]).item()
#     return max_id


def search_subimg(img, dtype, clip_model, clip_preprocessor, threshold=0.85, min_gap=0.03):
    '''
    根据类型查找与输入图像最相似的图像
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, clip_preprocessor = load_from_name('ViT-B-16', device=device, download_root='model/cn-clip/')
    inputs = clip_preprocessor(img).unsqueeze(0).to(device)
    img_feature = clip_model.encode_image(inputs)  # 提取输入图像特征

    candidate_feature = torch.load(f'data/{dtype}.pkl')  # 加载对应类型的特征向量数据

    img_feature /= img_feature.norm(dim=-1, keepdim=True)  # 归一化
    img_feature = img_feature.squeeze(dim=0)

    # 计算所有相似度并排序
    similarities = []
    for name, feature in candidate_feature.items():
        similarity = torch.dot(img_feature, feature).item()
        similarities.append((name, similarity))
    
    similarities.sort(key=lambda x: -x[1])  # 降序排列
    
    if not similarities:
        return 'Unknown'
    
    best_name, best_score = similarities[0]

    # 相似度阈值
    if best_score < threshold:
        return 'Unknown'
    
    # 区分度检查
    if len(similarities) > 1:
        second_score = similarities[1][1]
        gap = best_score - second_score
        
        if gap < min_gap:  # 与第二名太接近，拒绝
            return 'Unknown'
    
    return best_name


# def search_subimg(img, dtype, clip_model, clip_preprocessor, threshold=0.85, min_gap=0.03,
#                   verbose=False, return_detail=False):
#     '''
#     根据类型查找与输入图像最相似的图像 - 带动态Gap策略
    
#     参数:
#         verbose: 是否打印详细日志
#         return_detail: 是否返回详细信息（字典格式）
#     '''
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     inputs = clip_preprocessor(img).unsqueeze(0).to(device)
#     img_feature = clip_model.encode_image(inputs)

#     candidate_feature = torch.load(f'data/{dtype}.pkl')

#     img_feature /= img_feature.norm(dim=-1, keepdim=True)
#     img_feature = img_feature.squeeze(dim=0)

#     # 计算所有相似度并排序
#     similarities = []
#     for name, feature in candidate_feature.items():
#         similarity = torch.dot(img_feature, feature).item()
#         similarities.append((name, similarity))
    
#     similarities.sort(key=lambda x: -x[1])  # 降序排列
    
#     if not similarities:
#         if verbose:
#             print(f"  [{dtype.upper()}] 无候选特征")
#         if return_detail:
#             return {
#                 'name': 'Unknown',
#                 'reason': 'no_candidates',
#                 'score': None,
#                 'gap': None
#             }
#         return 'Unknown'
    
#     best_name, best_score = similarities[0]
    
#     # 收集Top5信息
#     top5 = [{'name': name, 'score': score} for name, score in similarities[:5]]
    
#     if verbose:
#         print(f"  [{dtype.upper()}] Top5相似度:")
#         for i, item in enumerate(top5, 1):
#             marker = "✓" if i == 1 else " "
#             print(f"    {marker} #{i}: {item['name']:20s} 分数={item['score']:.4f}")
    
#     # 计算区分度
#     gap = None
#     second_name = None
#     second_score = None
#     if len(similarities) > 1:
#         second_name, second_score = similarities[1]
#         gap = best_score - second_score
        
#         if verbose:
#             print(f"  [{dtype.upper()}] 区分度 Gap = {gap:.4f} (第二名: {second_name})")
    
#     # 阈值检查
#     passed_threshold = best_score >= threshold
    
#     if verbose:
#         print(f"  [{dtype.upper()}] 分数检查: {best_score:.4f} >= {threshold}? {passed_threshold}")
    
#     # ========== 动态 Gap 策略 ==========
#     if passed_threshold:
#         # 根据分数动态调整gap阈值（分数越高，要求的gap越小）
#         if best_score >= 0.90:           # 非常确定（分数很高）
#             effective_gap = min_gap * 0.5
#             confidence_level = "非常确定"
#         elif best_score >= 0.88:         # 比较确定
#             effective_gap = min_gap * 0.7
#             confidence_level = "比较确定"
#         elif best_score >= 0.86:         # 一般确定
#             effective_gap = min_gap * 0.85
#             confidence_level = "一般确定"
#         else:                            # 勉强通过
#             effective_gap = min_gap * 1.0
#             confidence_level = "勉强通过"
        
#         if verbose:
#             print(f"  [{dtype.upper()}] 置信度: {confidence_level} (分数={best_score:.4f})")
#             print(f"  [{dtype.upper()}] 动态Gap阈值: {effective_gap:.4f} (原始={min_gap:.4f}, 系数={effective_gap/min_gap:.2f})")
#     else:
#         # 如果连分数阈值都没通过，gap就无所谓了
#         effective_gap = min_gap
    
#     passed_gap = (gap is None) or (gap >= effective_gap)
    
#     if verbose and gap is not None:
#         print(f"  [{dtype.upper()}] Gap检查: {gap:.4f} >= {effective_gap:.4f}? {passed_gap}")
    
#     # 判定结果
#     if not passed_threshold:
#         reason = 'score_too_low'
#         result_name = 'Unknown'
#         if verbose:
#             print(f"  [{dtype.upper()}] ✗ 分数过低 ({best_score:.4f} < {threshold})")
#     elif not passed_gap:
#         reason = 'gap_too_small'
#         result_name = 'Unknown'
#         if verbose:
#             print(f"  [{dtype.upper()}] ✗ 区分度不足 ({gap:.4f} < {effective_gap:.4f})")
#     else:
#         reason = 'success'
#         result_name = best_name
#         if verbose:
#             print(f"  [{dtype.upper()}] ✓ 识别成功: {best_name}")
    
#     # 返回详细信息或仅返回名字
#     if return_detail:
#         return {
#             'name': result_name,
#             'reason': reason,
#             'score': best_score,  # 注意：分数越高越好
#             'gap': gap,
#             'threshold': threshold,
#             'min_gap': min_gap,
#             'effective_gap': effective_gap if passed_threshold else None,  # 新增：实际使用的gap阈值
#             'confidence_level': confidence_level if passed_threshold else None,  # 新增：置信度等级
#             'passed_threshold': passed_threshold,
#             'passed_gap': passed_gap,
#             'top5': top5,
#             'second_name': second_name,
#             'second_score': second_score
#         }
    
#     return result_name


def cal_tf(source, sentence):
    '''
    计算tf值
    '''
    tf = []
    for i in sentence:
        times = len(re.findall(source, i))
        tf.append((times * len(source)) / len(i))
    return tf


def cal_idf(source, sentence):
    '''
    计算idf值
    '''
    idf = []
    count = 0
    for i in sentence:
        if source in i:
            count += 1
    return math.log2(len(sentence) / (count+1))


def tf_idf_sort(keys, texts):
    '''
    找出与关键词的tfidf值最高的文本
    '''
    texts_score = [0 for i in range(0, 3000)]
    idfs = []
    # 遍历每个关键词
    for i in keys:
        idf = cal_idf(i, texts)  # 计算tf值
        tfs = cal_tf(i, texts)  # 计算idf值
        for j in range(0, len(texts)):
            # 对于每条文本，累加其该关键词的tfidf值到总得分中
            texts_score[j] += idf * tfs[j]
    fin_texts = [(texts[i], texts_score[i]) for i in range(0, len(texts))]  # 打包文本与其得分
    fin_texts = sorted(fin_texts, key=lambda x:x[1], reverse=True)  # 排序
    return [fin_texts[i][0] for i in range(0, min(20, len(fin_texts)))]  # 取前20条文本


def call_translator(text: str):
    '''
    调用翻译API（->中文）
    '''
    url = 'http://127.0.0.1:14800/trans2zh'
    payload = {'text': text}
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    return response.json().get('translation', '')


def seconds_to_hms(seconds):
    """
    将秒转换为 小时:分钟:秒.毫秒 格式
    """
    # 1. 先四舍五入转换成总毫秒数 (整数)
    total_milliseconds = int(round(seconds * 1000))
    # 2. 计算毫秒部分
    millisecs = total_milliseconds % 1000
    # 3. 计算总秒数，用于后续计算时分秒
    total_seconds = total_milliseconds // 1000
    # 4. 计算时、分、秒
    secs = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    # 返回格式: HH:MM:SS.mmm
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"


def call_qwen_smart(query, elements):
    """智能摘要统一入口"""
    import re
    import time
    
    MAX_CHUNK_SIZE = 2000
    MIN_SUMMARY_SIZE = 1500
    
    print(f"\n{'='*60}")
    print(f"[智能摘要] 收到文本: {len(query)}字")
    print(f"[智能摘要] 视觉元素: {len(elements) if elements else 0}个")
    print(f"{'='*60}\n")
    
    # ============ 策略1: 短文本 ============
    if len(query) <= MAX_CHUNK_SIZE:
        print("[策略] 文本较短，直接生成完整报告")
        
        result = call_qwen([query, elements], 'generate_summary')  # 🔥 直接返回字符串
        
        print(f"\n[完成] 最终报告生成成功，长度: {len(result)}字")
        print(f"[完成] 最终报告生成成功，内容: {result}\n")
        return result
    
    # ============ 策略2: 长文本 ============
    print("[策略] 文本过长，启动递归压缩")
    print(f"   - 原始长度: {len(query)}字")
    print(f"   - 目标长度: ≤{MIN_SUMMARY_SIZE}字")
    print(f"   - 分块大小: {MAX_CHUNK_SIZE}字/块")
    print(f"   - 处理流程: 分块 → 提取关键信息 → 递归压缩 → 生成报告\n")
    
    def split_into_chunks(text, max_size):
        sentences = re.split(r'([。！？\n])', text)
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            
            if len(current_chunk) + len(sentence) > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def compress_recursively(text, level=1):
        indent = "  " * level
        print(f"{indent}[第{level}层压缩]")
        print(f"{indent}├─ 输入长度: {len(text)}字")
        
        if len(text) <= MIN_SUMMARY_SIZE:
            print(f"{indent}└─ 已压缩到目标长度，停止递归\n")
            return text
        
        chunks = split_into_chunks(text, MAX_CHUNK_SIZE)
        print(f"{indent}├─ 分块数量: {len(chunks)}块")
        
        chunk_summaries = []
        
        for idx, chunk in enumerate(chunks):
            chunk_len = len(chunk)
            chunk_tokens = int(chunk_len * 1.75)
            print(f"{indent}├─ 处理块 {idx+1}/{len(chunks)} ({chunk_len}字 ≈ {chunk_tokens} tokens)...", end=" ")
            
            try:
                # 直接返回字符串
                summary = call_qwen([chunk, []], 'generate_summary_chunk')
                
                if summary:
                    chunk_summaries.append(summary.strip())
                    print(f"({len(summary)}字)")
                else:
                    fallback = chunk[:200]
                    chunk_summaries.append(fallback)
                    print(f"降级({len(fallback)}字)")
                    
            except Exception as e:
                print(f"失败: {str(e)[:50]}")
                fallback = chunk[:200]
                chunk_summaries.append(fallback)
            
            time.sleep(0.5)
        
        if not chunk_summaries:
            print(f"{indent}└─ 所有块处理失败，使用降级方案\n")
            return text[:MIN_SUMMARY_SIZE] 
        
        combined = "\n".join(chunk_summaries)
        print(f"{indent}├─ 合并后长度: {len(combined)}字")
        
        if len(combined) > MIN_SUMMARY_SIZE:
            print(f"{indent}└─ 仍超过目标长度，进入下一层递归\n")
            return compress_recursively(combined, level + 1)
        else:
            print(f"{indent}└─ 已压缩到目标长度\n")
            return combined
    
    try:
        compressed_text = compress_recursively(query, level=1)
        
        print(f"{'='*60}")
        print(f"[压缩完成]")
        print(f"   - 原始长度: {len(query)}字")
        print(f"   - 压缩后长度: {len(compressed_text)}字")
        print(f"   - 压缩率: {len(compressed_text)/len(query)*100:.1f}%")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"[压缩失败] {e}")
        print(f"   使用降级方案: 截取前{MIN_SUMMARY_SIZE}字\n")
        compressed_text = query[:MIN_SUMMARY_SIZE]
    
    print("[生成报告] 使用压缩后的文本 + 视觉元素")
    
    # 直接返回字符串
    final_result = call_qwen([compressed_text, elements], 'generate_summary')
    
    print(f"\n[完成] 最终报告生成成功，长度: {len(final_result)}字")
    print(f"[完成] 最终报告生成成功，内容: {final_result}\n")
    
    return final_result