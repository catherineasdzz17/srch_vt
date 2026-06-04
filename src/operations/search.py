import sys
sys.path.append("..")

import difflib
import torch
from transformers import pipeline

from config import DEFAULT_TABLE
from logs import LOGGER


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 预存储的特征向量库
img_feature = torch.load("data/image_feature_clip_v3.pkl")
text_feature = torch.load("data/text_feature.pkl")
clip_text_feature = torch.load("data/clip_text_feature.pkl")
graph_feature = torch.load("data/graph_feature.pkl")
mpnet_text_feature = torch.load("data/text_mpnet_feature.pkl")
graph_nation_feature = torch.load("data/Nation.pkl")
graph_politicians_feature = torch.load("data/Politicians.pkl")
print(f"type of img feature is {type(img_feature)}")


def find_similar_tensors(input_tensor, tensor_dict, k):
    '''
    在一组张量中查找与输入张量最相似的k个张量
    '''
    similarities = []
    key_list = []
    target_tensor_list = []

    # 遍历输入张量字典
    for key in tensor_dict.keys():
        key_list.append(key)
        if isinstance(tensor_dict[key], torch.Tensor):
            target_tensor_list.append(tensor_dict[key].squeeze().tolist())  # 转为list格式
        else:
            target_tensor_list.append(tensor_dict[key])

    target_tensor = torch.Tensor(target_tensor_list).to(device)  # 构建目标张量矩阵
    input_tensor = input_tensor.to(device)
    similarity = torch.sum(input_tensor * target_tensor, dim=-1)  # 计算相似度

    _, max_indices = torch.topk(similarity, k)  # 获取相似度最高的前k个索引

    # 根据索引还原对应的key
    top_k_keys = []
    for index in max_indices:
        top_k_keys.append(key_list[index])

    return top_k_keys


def mulit_search(input_tensor, k, search_type = "img"):
    '''
    多源向量相似度检索函数
    可根据不同类型在对应特征库中查找与输入向量最相似的前k项
    '''
    if search_type == "text":
        tensor_dict = text_feature
    elif search_type == "img":
        tensor_dict = img_feature
    elif search_type == "graph":
        tensor_dict = graph_feature
    elif search_type == 'mpnet_text':
        tensor_dict = mpnet_text_feature
    elif search_type == 'nation':
        tensor_dict = graph_nation_feature
    elif search_type == 'politicians':
        tensor_dict = graph_politicians_feature
    elif search_type == 'clip_text':
        tensor_dict = clip_text_feature

    # 执行查找函数
    key_list = find_similar_tensors(input_tensor, tensor_dict, k)

    return key_list


def search_in_milvus(table_name, query_sentence, top_k, model, milvus_cli, mysql_cli):
    '''
    基于Milvus的文本向量相似检索
    '''
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        vectors = model.sentence_encode([query_sentence])  # 将查询文本编码为向量
        LOGGER.info("Successfully insert query list")
        results = milvus_cli.search_vectors(table_name, vectors, top_k)  # 调用Milvus查询向量
        vids = [str(x.id) for x in results[0]]
        print("-----------------", vids)
        ids, title, text= mysql_cli.search_by_milvus_ids(vids, table_name)  # 从MySQL获取实际数据
        distances = [x.distance for x in results[0]]  # 提取向量相似度得分
        return ids, title, text, distances
    except Exception as e:
        LOGGER.error(f" Error with search : {e}")
        sys.exit(1)


def do_search(table_name, img_path, top_k, model, milvus_client, mysql_cli):
    '''
    基于Milvus的图像向量相似检索
    '''
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = model.resnet50_extract_feat(img_path)  # 提取图像的特征向量
        vectors = milvus_client.search_vectors(table_name, [feat], top_k)  # 调用Milvus查询向量
        vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_img_ids(vids, table_name)  # 从MySQL获取实际数据
        distances = [x.distance for x in vectors[0]]  # 提取向量相似度得分
        return paths, distances
    except Exception as e:
        LOGGER.error(f"Error with search : {e}")
        sys.exit(1)


def compute_keywords_score(sources, targets):
    '''
    计算两个关键词列表的相似度得分
    '''
    score = 0.0
    if len(targets) > 0:  # 非空计算得分
        for source in sources:
            # 在targets中查找与source最匹配的一个内容
            res = difflib.get_close_matches(source, targets, 1, cutoff=0.5)

            if len(res) == 0:
                res = ''
            else:
                res = res[0]
            # 计算两个字符串的相似度并累加
            score += difflib.SequenceMatcher(None, source, res).ratio()
    return score


def compute_keywords_score_and_tags(sources, targets):
    '''
    计算关键词相似度得分，并返回与每个source最接近的target
    '''
    score = 0.0
    new_sources = []
    if len(targets) > 0:  # 非空计算得分
        for source in sources:
            # 在targets中查找与source最匹配的一个内容
            res = difflib.get_close_matches(source, targets, 1, cutoff=0.5)
            if len(res) == 0:
                res = ''
            else:
                res = res[0]
                new_sources.append(res)
            # 计算两个字符串的相似度并累加
            score += difflib.SequenceMatcher(None, source, res).ratio()
    return score, new_sources


def get_entity(str0):
    '''
    从输入文本中提取命名实体
    '''
    # 命名实体识别
    pipe = pipeline('token-classification', model='src/model/model_ner')
    ans = pipe(str0)
    per = ['']  # 人物
    loc = ['']  # 地点
    org = ['']  # 组织
    for i in ans:
        cls = i['entity'][2:5]  # 实体类型（LOC/ORG/PER）
        lo = i['entity'][0:1]  # BIO标签中的B/I
        # 匹配实体类别
        if cls == 'LOC':
            if lo == 'B':
                loc.append(i['word'])
            else:
                loc[-1] = loc[-1] + i['word']
        elif cls == 'ORG':
            if lo == 'B':
                org.append(i['word'])
            else:
                org[-1] = org[-1] + i['word']
        else:
            if lo == 'B':
                per.append(i['word'])
            else:
                per[-1] = per[-1] + i['word']
    
    # 去重与合并 
    org = list(set(loc + org))
    per = list(set(per))
    return per, org