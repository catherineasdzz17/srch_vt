import torch
import face_recognition
import numpy as np
from collections import Counter
from cn_clip import clip
from PIL import Image

known_face='data/faces_v6.pkl'
known_face_encoding, known_name = torch.load(known_face)

# 1. 基本信息
print(f"文件加载成功")
print(f"   总特征向量数: {len(known_face_encoding)}")
print(f"   特征维度: {len(known_face_encoding[0])}")
print(f"   人物数量: {len(set(known_name))}")

# 2. 每个人的样本数
name_count = Counter(known_name)
print(f"\n人物样本统计:")
print(f"   {'人物':<15s} | 样本数")
print(f"   {'-'*30}")
for name, count in name_count.most_common():
    bar = '█' * min(count, 30)
    print(f"   {name:<15s} | {count:>3d} {bar}")


known_flag='data/flag.pkl'
flag_dict = torch.load(known_flag)

# 1. 基本信息
print(f"文件加载成功")
print(f"   数据格式: 字典 (国家名 -> 特征向量)")
print(f"   旗帜数量: {len(flag_dict)}")

# 提取数据
flag_labels = list(flag_dict.keys())
flag_features = torch.stack(list(flag_dict.values())).to('cuda')

print(f"   特征维度: {flag_features[0].shape}")

# 2. 旗帜分类统计
print(f"\n旗帜分类:")

# 按类别分组
countries = []
organizations = []
regions = []

org_keywords = ['联合国', '欧盟', '北约', '国协', '东盟', '阿盟', '组织', '共同体', '协会', '联邦', '北大西洋', '东南亚']
region_keywords = ['香港', '澳门', '英格兰', '苏格兰', '威尔士']

for label in flag_labels:
    if any(org in label for org in org_keywords):
        organizations.append(label)
    elif any(reg in label for reg in region_keywords):
        regions.append(label)
    else:
        countries.append(label)

print(f"   国家: {len(countries)} 个")
print(f"   国际组织: {len(organizations)} 个")
print(f"   地区: {len(regions)} 个")

# 3. 显示旗帜列表
print(f"\n旗帜列表:")

if countries:
    print(f"\n   国家 ({len(countries)}个):")
    for i in range(0, len(countries), 6):
        print(f"   {', '.join(countries[i:i+6])}")

if organizations:
    print(f"\n   国际组织 ({len(organizations)}个):")
    for org in organizations:
        print(f"   - {org}")

if regions:
    print(f"\n   地区 ({len(regions)}个):")
    for reg in regions:
        print(f"   - {reg}")

def format_list(lst):
    return '[' + ', '.join(f"'{x}'" for x in lst) + ']'

person_list = sorted(list(set(known_name)))
print("\n人物列表（person）:")
print(format_list(person_list))

print("\n国家列表（flag）:")
print(format_list(countries))