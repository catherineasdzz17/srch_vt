# import heapq
import torch
# import zhconv
# import pandas as pd

# from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import BGEM3FlagModel


maxl = 250000


def encode(sentence=None, model=None, return_dense=True, return_sparse=True, return_colbert_vecs=True):
    '''
    编码输入文本
    输出是一个字典，包含三种编码结果
        dense_vecs：密集编码，将文本转成一个编码
        lexical_weights：bm25后每个词的值，需要model来计算文档相似度
        colbert_vecs：colbert编码
    '''
    if isinstance(sentence, str):
        sentence = [sentence]  # 单个字符串则转成只有一个元素的列表
    # 编码
    output = model.encode(sentence, return_dense=return_dense, return_sparse=return_sparse, return_colbert_vecs=return_colbert_vecs, max_length=2048)
    return output


def cal_similarity(a, b, model=None, stype='dense'):
    '''
    计算两个句子的相似度
    '''
    if stype == 'dense':  # 使用向量内积计算dense_vecs的相似度
        return a['dense_vecs'] @ b['dense_vecs'].T
    elif stype == 'sparse':  # 计算lexical_weights的相似度
        return model.compute_lexical_matching_score(a['lexical_weights'], b['lexical_weights'])
    else:  # 计算colbert_vecs的相似度
        return model.colbert_score(a['colbert_vecs'], b['colbert_vecs'])


# def search_text(query, model=None, candidate_set=None):
#     '''
#     检索与query最相似的文本
#     '''
#     min_heap = []
#     query_embed = encode(sentence=query, model=model, return_colbert_vecs=False)  # 编码查询向量
#     '''
#     若待检测包为纯文本，eve为元组，第一项是编码结果，结构为字典，包含dense_vecs和lexical_weights两个key，第二项是文本
#     若待检测包为视频文本，eve为元组，第一项是编码结果，结构为字典，包含dense_vecs和lexical_weights两个key,第二项是视频路径，第三项是视频文本
#     若待检测包为图片文本，eve为元组，第一项是编码结果，结构为字典，包含dense_vecs和lexical_weights两个key,第二项是图片路径，第三项是图片文本
#     '''
#     # 遍历待检测包的每一项
#     for eve in candidate_set:
#         bm25 = cal_similarity(query_embed, eve[0], model=model, stype='sparse')[0][0]  # 计算bm25相似度
#         # if len(min_heap) < 10:
#         #     heapq.heappush(min_heap, (bm25, eve[0]))
#         # else:
#         #     # 如果当前相似度大于堆顶的相似度，替换堆顶元素
#         #     if bm25 > min_heap[0][0]:
#         #         heapq.heapreplace(min_heap, (bm25, eve[0]))
#         if len(eve)==2:  # （编码，文本）
#             min_heap.append((bm25, eve[1]))  # 只保留文本
#         else:  # （编码，路径，描述）
#             min_heap.append((bm25, eve[1], eve[2]))  # 保留路径和描述
#     min_heap = sorted(min_heap, reverse=True, key=lambda x:x[0])[:10]  # 按相似度排序
#     # final_sque = []
#     # for eve in min_heap:
#     #     sim = cal_similarity(ans,eve[2],stype='sparse')
#     #     final_sque.append((sim,eve[1]))
#     return min_heap
def search_text(query, model=None, candidate_set=None, threshold=0.4):
    '''
    检索与query最相似的文本
    '''
    min_heap = []
    query_embed = encode(sentence=query, model=model, return_colbert_vecs=False)  # 编码查询向量
    
    for eve in candidate_set:
        bm25 = cal_similarity(query_embed, eve[0], model=model, stype='sparse')[0][0]  # 计算bm25相似度
        
        # 根据是否启用阈值过滤来决定是否添加结果
        if bm25 > threshold:
            if len(eve) == 2:  # （编码，文本）
                min_heap.append((bm25, eve[1]))
            else:  # （编码，路径，描述）
                min_heap.append((bm25, eve[1], eve[2]))
    
    min_heap = sorted(min_heap, reverse=True, key=lambda x: x[0])[:10]
    return min_heap


def embed2sparse_matrix(embed):
    '''
    将编码后的稀疏向量转换为稀疏矩阵格式
    '''
    embed = dict(embed['lexical_weights'][0])
    query_keys = list(map(int, embed.keys()))
    query_values = list(embed.values())
    query_matrix = csr_matrix((query_values, ([0] * len(query_values), query_keys)), shape=(1, maxl))
    return query_matrix


def build_sparse_matrix(tmp):
    '''
    将多个稀疏向量构建成一个稀疏矩阵
    '''
    shape = (len(tmp), maxl)
    rows = []
    cols = []
    values = []
    texts = []

    # 遍历文本和对应编码表示
    for i, (text, embed) in enumerate(tmp):
        embed = dict(embed['lexical_weights'][0])
        texts.append(text)
        for key, value in embed.items():
            rows.append(i)
            cols.append(int(key))
            values.append(value)

    sparse_matrix = csr_matrix((values, (rows, cols)), shape=shape)
    return texts, sparse_matrix


def cal_bm25_similarity(query, model=None, candidate_set=None):
    '''
    计算给定查询和候选集合之间的BM25相似度
    '''
    query_embed = encode(sentence=query, model=model, return_colbert_vecs=False)  # 编码查询文本
    tmp = torch.load(f'data/{candidate_set}')  # 加载候选集合
    texts, document_matrix = build_sparse_matrix(tmp=tmp)  # 构建候选集合的稀疏矩阵
    query_matrix = embed2sparse_matrix(query_embed)  # 构建查询文本的稀疏矩阵
    similarities = cosine_similarity(query_matrix, document_matrix).flatten()  # 计算余弦相似度
    sorted_indices = similarities.argsort()[::-1][:10]  # 排序，取前10条
    ans = []
    for i in sorted_indices:
        ans.append((i,texts[i]))
    return ans


if __name__=='__main__':
    model = BGEM3FlagModel('model/bge-m3', use_fp16=True)
    cal_bm25_similarity(query='泽连斯基', model=model, candidate_set='text_data.pkl')

    # final_ans = []
    # with torch.no_grad():
    #     model = BGEM3FlagModel('model/bge-m3', use_fp16=True)
    #     df = pd.read_csv('文本测试集.csv')
    #     # ans = []
    #     for index,row in tqdm(df.iterrows(), total=len(df)):
    #         # text = zhconv.convert(row['text'],'zh-hans')
    #         text = row['text']
    #         # output = encode(sentence=text,model=model,return_colbert_vecs=False)
    #         # print(text)
    #         ans = search_video_text(text, model=model)
    #         dic = {}
    #         dic['text'] = text
    #         for i in range(0,10):
    #             dic[f'text{i}'] = ans[i][1]

    # def format_dict(d):
    #     return ",\n".join([f'  "{k}": {json.dumps(v)}' for k, v in d.items()])

    # # 定义一个函数来格式化整个列表
    # def format_list(lst):
    #     formatted_items = [f"{{\n{format_dict(item)}\n}}" for item in lst]
    #     return "[\n" + ",\n".join(formatted_items) + "\n]"

    # # 格式化数据
    # formatted_data = format_list(final_ans)

    # # 写入文件
    # with open('output.json', 'w') as f:
    #     f.write(formatted_data)