import copy
import difflib
import json
import os
import re


class SearchGraph:
    def __init__(self, path=None):
        # 加载图谱数据
        with open(os.path.join(path, '图谱微博节点1333-增加一跳.json'), 'r', encoding='utf8') as f:
            self.entities = json.load(f)

        with open(os.path.join(path, '图谱关系总.json'), 'r', encoding='utf8') as ff:
            self.relations = json.load(ff)

        # 初始化映射字典
        self.id2name = {}
        self.name2id = {}
        self.person_name2id = {}
        # 增加关系type的中文映射
        self.relation_type_to_cn = {
            'Image_Of': '图片',
            '国旗': '国旗',
            'Nationality': '国籍',
            'Served_As': '职位',
            '别名': '别名'
        }
        # 增加一些网络流行别称到标准人名的映射
        self.person2person = {
            '乔·拜登上议程': '约瑟夫·拜登',
            '乔·拜登': '约瑟夫·拜登',
            '川建国': '特朗普',
            '川建国同志': '特朗普',
            '懂王': '特朗普',
            '乔拜登': '约瑟夫·拜登',
            '瞌睡乔': '约瑟夫·拜登',

        }
        # 遍历实体，构建字典映射
        for entity in self.entities:
            if entity['labels'][0] == 'Politicians':
                self.person_name2id[entity['name']] = entity['id']
            self.id2name[entity['id']] = entity
            self.name2id[entity['name']] = entity['id']


    def find_entities_in_graph(self, name):
        '''
        在图谱中查找给定名称的实体
        输入：二元组（类型，名称）
        '''
        ans = []
        label, name = name
        # 处理别称映射
        if name in self.person2person.keys():
            name = self.person2person[name]
        if '人物' in label:
            # 人物在人物表中
            if name in self.person_name2id.keys():
                ans.append(name)
            else:
                # 否则采用模糊匹配，从已有中找到最相似的
                ans.extend(difflib.get_close_matches(name, self.person_name2id.keys(), 1, cutoff=0.5))
        else:
            # 使用所有实体名称的映射表来匹配
            if name in self.name2id.keys():
                ans.append(name)
            else:
                # 否则模糊匹配
                ans.extend(difflib.get_close_matches(name, self.name2id.keys(), 1, cutoff=0.5))
        return ans


    def find_one_step_neighbors(self, name):
        '''
        查找某个实体的一跳邻居（直连节点）
        '''
        id = self.name2id[name]  # 获取实体的唯一id
        neighbors = []
        # 遍历所有关系
        for rel in self.relations:
            # 查询与输入实体相连的边
            if rel['sourceId'] == id:
                if rel['targetId'] in self.id2name.keys():  # 确保目标节点存在
                    target = copy.deepcopy(self.id2name[rel['targetId']])  # 创建目标节点的副本
                    target['labels'][0] = self.relation_type_to_cn[rel['type']]  # 替换为中文标签
                    neighbors.append(target)
                    # 国旗类则进一步查找其一跳邻居
                    if target['labels'][0] == '国旗':
                        neighbors.extend(self.find_one_step_neighbors(target['name']))
            # if rel['targetId'] == id:
            #     if rel['sourceId'] in self.id2name.keys():
            #         neighbors.append(self.id2name[rel['sourceId']])
        return neighbors


    def process_response(self, response: str):
        '''
        提取结构化二元组
        '''
        names = []
        values = []
        response = response.strip()
        response = response.replace('\n', '')
        # 使用中英文分号分割整个字符串为key : value
        key_values = re.split(';|；', response)
        # 解析每个key : value
        for key_value in key_values:
            kv = re.split(':|：', key_value.strip())
            if len(kv) > 1:
                values.append((kv[0].strip(), kv[1].strip()))

        # 拆分多个实体
        for k, val in values:
            vs = re.split('，|、|,', val)
            for v in vs:
                # 过滤无效内容
                if len(v) > 0 and '无' not in v and '无明显' not in v and '无明确' not in v and '无提及' not in v and '未提及' not in v:
                    names.append((k, v.strip()))
        return names


    def search(self, name):
        '''
        基于名称查询图谱中的实体及其一跳邻居
        '''
        ans = []
        # 查找实体
        root = self.find_entities_in_graph(name)
        if len(root) > 0:
            # 获取id
            rootId = self.name2id[root[0]]
            ans.append(self.id2name[rootId])
            # 查找邻居
            neighbors = self.find_one_step_neighbors(root[0])
            ans.extend(neighbors)
        print(ans)
        return ans