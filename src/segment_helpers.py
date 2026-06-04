import copy
import cv2
import os
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from config import SAM_CHECKPOINT, SAM_TYPE


class Segment_generator:
    def __init__(self):
        # image_path = './weibo_images/img/547.jpg'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('sam_device: ', device)

        # 加载SAM模型
        sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
        _ = sam.to(device=device)

        # 实例化SAM自动掩码生成器
        mask_generator = SamAutomaticMaskGenerator(sam)
        self.mask_generator = mask_generator


    def get_model(self):
        return self.mask_generator


def apply_mask(image, mask, alpha_channel=True):  
    '''
    将掩码应用于图像上
    '''
    if alpha_channel:  # 启用透明通道
        alpha = np.zeros_like(image[..., 0])  # 制作掩体，与图像shape相同的全黑通道
        alpha[mask == 1] = 255  # 兴趣地方标记为1，且为白色
        image = cv2.merge((image[..., 0], image[..., 1], image[..., 2], alpha))  # 合并图像，返回一个4通道图像
    else:
        image = np.where(mask[..., None] == 1, image, 0)  # 将非掩码部分置为黑色，返回3通道图像
    return image


def mask_image(image, mask, crop_mode_=True):
    '''
    提取掩码覆盖的目标区域
    '''
    if crop_mode_:  # 裁剪图像为掩码的最小外接矩形
        y, x = np.where(mask)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
        cropped_image = image[y_min : y_max + 1, x_min : x_max + 1]
        # 应用mask
        masked_image = apply_mask(cropped_image, cropped_mask)
    else:
        masked_image = apply_mask(image, mask)
    return masked_image


def save_masked_image(image, filepath):
    '''
    保存掩码图像
    '''
    if image.shape[-1] == 4:  # 通道数为4
        cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # png格式保存并压缩
    else:
        cv2.imwrite(filepath, image)
    print(f"Saved as {filepath}")


def save_anns(anns, image, path):
    '''
    获取并保存分割掩码图像
    '''
    if len(anns) == 0:
        return
    sorted_anns = select_anns(anns)  # 筛选掩码
    index = 1

    for ann in sorted_anns:
        mask_2d = ann['segmentation']  # 获取二值掩码图
        segment_image = copy.copy(image)
        masked_image = mask_image(segment_image, mask_2d)  # 提取图像
        filename = str(index) + '.png'
        filepath = os.path.join(path, filename)
        save_masked_image(masked_image, filepath)  # 保存图像
        index = index + 1


def select_anns(anns):
    '''
    筛选语义分割掩码
    '''
    # 按面积从大到小排序
    anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ans = []
    # 遍历所有掩码
    for i in anns:
        if len(ans) == 0:
            ans.append(i)  # 添加第一个掩码
            continue
        i_seg = i['segmentation']  # 提取掩码的二值图
        flag = True
        # 与其他掩码逐一比较
        for j in ans:
            j_seg = j['segmentation']
            same = np.count_nonzero(i_seg & j_seg)  # 按位与
            total = min(np.count_nonzero(i_seg), np.count_nonzero(j_seg))
            if same / total >= 0.2:  # 重叠超过20%则冗余
                flag = False
                break
        if flag:
            ans.append(i)
    # 最多返回5个掩码
    return ans[0 : 5]


def get_imgs(path):
    '''
    从指定文件夹中获取所有图片的完整路径和文件名列表
    '''
    pics = []  # 完整路径
    fs = []  # 文件名
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']) and not f.startswith('.DS_Store')):
            pics.append(os.path.join(path, f))
            fs.append(f)
    return pics, fs


def get_segment_images(model, source, save_path):
    '''
    将图片分割、筛选的方法
    输入：分割模型sam与图片image与保存图片的位置path
    输出：包含一系列分割后的rgba格式的图片
    '''
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if isinstance(source, str):  # 根据路径读取图像
        source = cv2.imread(source)
    image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)  # 转换为RGB
    if image is not None:
        masks = model.generate(image)  # 生成掩码
        save_anns(masks, source, save_path)  # 保存掩码图像


def get_segment_images_v2(model, image):
    '''
    将图片分割、筛选的方法
    输入：分割模型sam与图片image
    输出：包含一系列分割后的rgba格式的图片
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finans = []
    if image is not None:
        masks = model.generate(image)  # 生成掩码
        masks = select_anns(masks)  # 保存掩码图像
        # 获取掩码图像列表
        for mask in masks:
            mask_2d = mask['segmentation']
            segment_image = copy.copy(image)
            masked_image = mask_image(segment_image, mask_2d)
            finans.append(masked_image)
    return finans