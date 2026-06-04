import os
import sys
sys.path.append("..")

import json
import pandas as pd

from diskcache import Cache
from PIL import Image as im

from config import DEFAULT_TABLE
from logs import LOGGER


def extract_features(file_dir, model):
    '''
    提取文本的向量表示
    '''
    try:
        data = pd.read_csv(file_dir)
        title_data = data['text'].tolist()
        text_data = data['text'].tolist()
        # 文本编码
        sentence_embeddings = model.sentence_encode(title_data)
        return title_data, text_data, sentence_embeddings
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")
        sys.exit(1)


def format_data(ids, title_data, text_data):
    '''
    打包向量id和对应文本内容
    '''
    # Combine the id of the vector and the question data into a list
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), title_data[i], text_data[i])
        data.append(value)
    return data


def do_load(collection_name, file_dir, model, milvus_client, mysql_cli):
    '''
    将文本向量导入Milvus，将文本数据导入MySQL
    '''
    if not collection_name:
        collection_name = DEFAULT_TABLE
    # 提取文本和对应向量
    title_data, text_data, sentence_embeddings = extract_features(file_dir, model)
    # 插入向量到Milvus
    ids = milvus_client.insert(collection_name, sentence_embeddings)
    milvus_client.create_index(collection_name)

    # Load the collection data into RAM just AFTER creating index
    milvus_client.load(collection_name)
    # 创建MySQL表并写入数据
    mysql_cli.create_mysql_table(collection_name)
    mysql_cli.load_data_to_mysql(collection_name, format_data(ids, title_data, text_data))
    return len(ids)


def get_imgs(path):
    '''
    获取指定路径下所有图像文件路径
    '''
    pics = []
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']) and not f.startswith('.DS_Store')):
            pics.append(os.path.join(path, f))
    return pics


def extract_img_features(img_dir, model):
    '''
    提取图像的向量表示
    '''
    try:
        cache = Cache('./tmp')  # 创建一个缓存对象，用于记录进度
        feats = []  # 图像的特征向量
        names = []

        img_list = get_imgs(img_dir)  # 获取图像文件路径
        total = len(img_list)
        cache['total'] = total
        # 遍历图像提取特征
        for i, img_path in enumerate(img_list):
            try:
                norm_feat = model.resnet50_extract_feat(img_path)
                feats.append(norm_feat)
                names.append(img_path.encode())  # 编码文件路径
                cache['current'] = i + 1
                print(f"Extracting feature from image No. {i + 1} , {total} images in total")
            except Exception as e:
                LOGGER.error(f"Error with extracting feature from image {e}")
                continue
        return feats, names
    except Exception as e:
        LOGGER.error(f"Error with extracting feature from image {e}")
        sys.exit(1)


def format_img_data(ids, names):
    '''
    打包向量id和对应的图像路径
    '''
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data


def do_img_load(table_name, image_dir, model, milvus_client, mysql_cli):
    '''
    将图像向量导入Milvus，将图像数据导入MySQL
    '''
    if not table_name:
        table_name = DEFAULT_TABLE
    # 提取图像特征向量
    vectors, names = extract_img_features(image_dir, model)
    # 将向量写入Milvus
    ids = milvus_client.insert(table_name, vectors)
    milvus_client.create_index(table_name)
    # 创建MySQL表并写入图像数据
    mysql_cli.create_mysql_img_table(table_name)
    mysql_cli.load_data_to_img_mysql(table_name, format_img_data(ids, names))
    return len(ids)


def image_map(img_dir):
    '''
    从制定路径中加载图像数据并保存
    '''
    img_list = get_imgs(img_dir)
    map = {}
    count = 0
    for imgfile in img_list:
        img = im.open(imgfile)
        names = imgfile.split('.')
        print(names[-1])
        img.save('F:/PKU/RKGImageSource/new_test/' + str(count) + '.' + names[-1])
        count += 1
        map[str(count) + '.jpg'] = img
    return map


if __name__ == '__main__':
    # Make inference for images
    from encode import Resnet50
    model = Resnet50()
    image_feat_path = '../../data/test_img_feat.jsonl'
    # img_dir = 'F:/PKU/RKGImageSource/researchdemo'
    # map = image_map(img_dir)
    img_dir = 'F:/PKU/segment-anything/result'
    img_dir = 'F:/PKU/RKGImageSource/new_test'
    with open(image_feat_path, 'w') as fout:
        image_features, image_ids = extract_img_features(img_dir, model)
        for image_id, image_feature in zip(image_ids, image_features):
            # print(image_id.decode(), list(image_feature))
            fout.write("{}\n".format(json.dumps({"image_id": image_id.decode(), "feature": image_feature.tolist()})))