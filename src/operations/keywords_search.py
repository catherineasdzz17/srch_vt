import logging
import math
import os.path
import pickle
from collections import Counter

try:
    import pkuseg
except ModuleNotFoundError:
    import spacy_pkuseg as pkuseg


logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S",
                    level=logging.INFO)


class TFIDFSearch:
    def __init__(self, corpus=None, update=False, path=None):
        seg = pkuseg.pkuseg()  # 以默认配置加载默认中文分词器
        # 读取停用词数据
        stopwords = []
        with open(os.path.join(path, 'stopwords_Chinese.txt'), mode='r', encoding='utf-8') as f:
            for line in f:
                stopwords.append(line.strip())

        # 更新模型
        if update:
            self.corpus = corpus  # 待检索语料
            # idf计算
            self.idf_fuc = lambda word, count_lst: math.log(len(count_lst)) / (
                    1 + sum(1 for count in count_lst if word in count))
            # 分词、去除停用词
            self.delete_stops = lambda x: [i for i in seg.cut(x) if i not in stopwords]
            # id到文本的映射
            self.id2sentence = {}
            # 构建检索模型
            self.search_model = self.build_search()
            # 保存模型文件
            with open(os.path.join(path, 'id2sentence.pkl'), mode='wb') as f:
                pickle.dump(self.id2sentence, f)
                f.close()
            with open(os.path.join(path, 'search_model.pkl'), mode='wb') as ff:
                pickle.dump(self.search_model, ff)
                ff.close()
        # 否则直接加载
        else:
            self.corpus = corpus  # 待检索语料
            # idf计算
            self.idf_fuc = lambda word, count_lst: math.log(len(count_lst)) / (
                    1 + sum(1 for count in count_lst if word in count))
            self.delete_stops = lambda x: [i for i in seg.cut(x) if i not in stopwords]
            with open(os.path.join(path, 'id2sentence.pkl'), mode='rb') as f:
                self.id2sentence = pickle.load(f)
                f.close()
            with open(os.path.join(path, 'search_model.pkl'), mode='rb') as ff:
                self.search_model = pickle.load(ff)
                ff.close()


    def word2index(self):
        '''
        实现倒排索引
        '''
        count_lst = []  # Counter结果
        index_dic = {}  # 倒排索引结果
        tf_dic = {}  # tf结果
        # 遍历语料
        for index, sent in self.corpus.iterrows():
            # 文本处理与映射
            self.id2sentence[sent['id']] = sent['text']
            words = self.delete_stops(sent['text'])
            word_count_dic = Counter(words)  # 统计词频
            count_lst.append(word_count_dic)
            # 遍历每一个词
            for word in words:
                if not word.strip():
                    continue
                # 构建倒排索引
                index_dic.setdefault(word, set()).add(sent['id'])
                # 计算tf
                tf_dic.setdefault(word, word_count_dic[word] / len(words))
        return index_dic, count_lst, tf_dic


    def build_search(self):
        '''
        构建search模型
        '''
        index_dic, count_lst, tf_dic = self.word2index()
        result = {}
        tf_idf_dic = {}  # tfidf结果
        # 遍历词频统计
        for i, count_dic in enumerate(count_lst):
            # 遍历每一个词
            for word, count in count_dic.items():
                # 计算tfidf
                word_idf = self.idf_fuc(word, count_lst)
                word_tf = tf_dic.get(word)
                word_tf_idf = word_tf * word_idf
                logging.info("word={}\ttf-idf={}".format(word, word_tf_idf))
                tf_idf_dic.setdefault(word, word_tf_idf)
                # 构建search模型结构（tfidf和idx）
                result[word] = {"tfidf": word_tf_idf, "b2s": index_dic.get(word, {})}
        # print(sorted(result.items(), key=lambda x: float(x[1]['tfidf']), reverse=True))
        return result


    def search(self, search_sent, topk):
        '''
        搜索
        '''
        # 加载搜索模型和分词器
        search_model = self.search_model
        search_words = self.delete_stops(search_sent)
        res = {}
        # 遍历查询语句中每一个词
        for index, search_word in enumerate(search_words):
            tfidf_index = search_model.get(search_word, {})
            if tfidf_index:
                # 更新每个句子得分
                for idx in tfidf_index['b2s']:
                    res[idx] = res.get(idx, 0) + float(tfidf_index['tfidf'])
        # 按分数排序并取topk个
        ans = sorted(res.items(), key=lambda x: x[1], reverse=True)
        ans = ans[:topk]
        sentences = []
        for r in ans:
            # print("id=", r[0], "sentence=", self.id2sentence[r[0]], "score=", r[1])
            sentences.append(self.id2sentence[r[0]])  # 获取对应的文本内容
        return sentences