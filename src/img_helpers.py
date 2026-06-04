import os
import torch

from PIL import Image
from face_recg import face_recg
from my_utils import rgba2rgb, search_subimg


# def img_recg(processor, class_model, CLIP_MODEL, pic):
#     '''
#     图像元素识别
#     输入：相关模型组件与需要识别的图片路径（图片为rgba格式）
#     输出：识别后的结果字典
#     '''
#     element_dict = {
#         'img': '',  # 图片路径
#         'elements': [],  # 元素识别结果
#         'class': 0  # 类别：0人物、1旗帜、2武器装备、3其他
#     }

#     img = Image.open(pic)  # 加载图片
#     img = rgba2rgb(img)  # 转换格式
#     # 图像分类
#     inputs = processor(images=img, return_tensors="pt").to('cuda:0')
#     pred = class_model(inputs.data['pixel_values']).to('cuda:0')
#     pred_label = torch.argmax(pred).item()
#     key_words = []
#     # 元素识别及关键词提取
#     if True:
#         if pred_label == 0:
#             element_dict['img'] = os.path.join(os.getcwd(), pic)
#             person_name = face_recg(pic)  # 人脸识别
#             element_dict['elements'] = [person_name]
#             key_words.append(person_name)

#         elif pred_label == 1:
#             element_dict['img'] = os.path.join(os.getcwd(), pic)
#             org_name = search_subimg(img, 'flag', CLIP_MODEL.get_model(), CLIP_MODEL.get_preprocess())  # 旗帜识别
#             element_dict['elements'] = [org_name]
#             element_dict['class'] = 1
#             key_words.append(org_name)

#         elif pred_label == 2:
#             element_dict['img'] = os.path.join(os.getcwd(), pic)
#             element_dict['elements'] = ['武器装备']
#             element_dict['class'] = 2

#         else:
#             element_dict['elements'] = ['杂物']
#             element_dict['img'] = os.path.join(os.getcwd(), pic)
#             element_dict['class'] = 3
#     return element_dict, key_words


def img_recg(processor, class_model, CLIP_MODEL, pic,
             face_threshold=0.45, face_gap=0.08, 
             flag_threshold=0.75, flag_gap=0.08):
    
    element_dict = {
        'img': '',
        'elements': [],
        'class': 0
    }

    img = Image.open(pic)
    img = rgba2rgb(img)
    
    inputs = processor(images=img, return_tensors="pt").to('cuda:0')
    pred = class_model(inputs.data['pixel_values']).to('cuda:0')
    pred_label = torch.argmax(pred).item()
    
    key_words = []
    
    if pred_label == 0:
        element_dict['img'] = os.path.join(os.getcwd(), pic)
        person_name = face_recg(pic, threshold=face_threshold, min_gap=face_gap)
        element_dict['elements'] = [person_name]  # ← 关键修复
        if person_name != 'Unknown':
            key_words.append(person_name)
    
    elif pred_label == 1:
        element_dict['img'] = os.path.join(os.getcwd(), pic)
        org_name = search_subimg(img, 'flag', 
                                CLIP_MODEL.get_model(), 
                                CLIP_MODEL.get_preprocess(),
                                threshold=flag_threshold, 
                                min_gap=flag_gap)
        element_dict['elements'] = [org_name]  # ← 关键修复
        element_dict['class'] = 1
        if org_name != 'Unknown':
            key_words.append(org_name)
    
    elif pred_label == 2:
        element_dict['img'] = os.path.join(os.getcwd(), pic)
        element_dict['elements'] = ['武器装备']
        element_dict['class'] = 2
    
    else:
        element_dict['elements'] = ['杂物']
        element_dict['img'] = os.path.join(os.getcwd(), pic)
        element_dict['class'] = 3
    
    return element_dict, key_words


def image_handler_main(input_image=None, ):
    pass