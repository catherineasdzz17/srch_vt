import os
import torch
import numpy as np
from PIL import Image, ImageEnhance
import albumentations as A
import face_recognition
from cn_clip.clip import load_from_name
import random
import warnings
import cv2

warnings.filterwarnings('ignore', category=UserWarning, module='PIL.Image')

# ==================== 图像预处理 ====================

def preprocess_image(img_path):
    """统一的图像预处理函数"""
    img = Image.open(img_path)
    
    if img.mode == 'P':
        if 'transparency' in img.info:
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')
    
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        return rgb_img
    
    if img.mode == 'LA':
        img = img.convert('RGBA')
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[1])
        return rgb_img
    
    if img.mode != 'RGB':
        return img.convert('RGB')
    
    return img


# ==================== 针对侧脸的增强 ====================

def augment_face_image_robust(img_pil, num_aug=30, ensure_face_visible=True):
    """
    专门针对侧脸的人脸增强
    
    关键改进：
    1. 增加更大的旋转角度（模拟侧脸）
    2. 增加3D透视变换（模拟不同视角）
    3. 减少遮挡（避免破坏关键特征）
    """
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    
    # ✅ 针对侧脸的增强策略
    spatial_aug = A.Compose([
        # 更大的旋转角度（模拟头部转动）
        A.Affine(
            rotate=(-30, 30),      # 增加到±30度
            scale=(0.85, 1.15),    # 更大的缩放范围
            shear=(-15, 15),       # 更大的错切
            p=0.8                  # 提高概率
        ),
        # 3D透视变换（模拟不同视角）
        A.Perspective(scale=(0.05, 0.20), p=0.6),  # 提高概率和强度
        
        # 减少遮挡（侧脸本身特征就少）
        A.CoarseDropout(
            max_holes=1,            # 只遮挡1个区域
            max_height=int(h*0.10), # 只遮挡10%
            max_width=int(w*0.10),
            min_holes=1,
            fill_value=128,
            p=0.2                   # 降低概率
        ),
    ])
    
    # ✅ 光照增强（模拟不同光照条件）
    color_aug = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ])
    
    augmented = [img_pil]
    valid_count = 1
    attempts = 0
    max_attempts = num_aug * 5  # 增加到5倍尝试
    
    while len(augmented) < num_aug + 1 and attempts < max_attempts:
        attempts += 1
        
        # 空间变换
        aug_np = spatial_aug(image=img_np)['image']
        
        # 光照变换
        aug_np = color_aug(image=aug_np)['image']
        
        aug_pil = Image.fromarray(aug_np)
        
        # 额外的PIL增强
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(aug_pil)
            aug_pil = enhancer.enhance(random.uniform(0.6, 1.4))
        
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(aug_pil)
            aug_pil = enhancer.enhance(random.uniform(0.7, 1.3))
        
        # 验证人脸可见性
        if ensure_face_visible:
            test_array = np.array(aug_pil)
            # ✅ 使用CNN模型检测（对侧脸更鲁棒）
            try:
                test_enc = face_recognition.face_encodings(test_array, model='large')
                if len(test_enc) > 0:
                    augmented.append(aug_pil)
                    valid_count += 1
            except:
                # 如果CNN失败，尝试HOG
                test_enc = face_recognition.face_encodings(test_array)
                if len(test_enc) > 0:
                    augmented.append(aug_pil)
                    valid_count += 1
        else:
            augmented.append(aug_pil)
    
    if ensure_face_visible and valid_count < num_aug + 1:
        print(f"      ⚠️ 生成了 {valid_count}/{num_aug+1} 个有效增强 (尝试 {attempts} 次)")
    
    return augmented


# ==================== 针对褶皱倾斜的旗帜增强 ====================

def augment_flag_image_robust(img_pil, num_aug=40):
    """
    专门针对褶皱、倾斜旗帜的增强
    
    关键改进：
    1. 增加弹性变形（模拟褶皱）
    2. 增加网格扭曲（模拟飘动）
    3. 增加更大的透视变换（模拟倾斜）
    4. 增加运动模糊（模拟飘动）
    """
    img_np = np.array(img_pil)
    
    # ✅ 专门针对褶皱和倾斜的增强
    spatial_aug = A.Compose([
        # 更强的透视变换（模拟各种角度）
        A.Perspective(scale=(0.05, 0.3), p=0.8),  # 提高强度和概率
        
        # 更大的仿射变换
        A.Affine(
            rotate=(-45, 45),      # 增加到±45度
            scale=(0.7, 1.3),      # 更大范围
            shear=(-20, 20),       # 更大错切
            p=0.9                  # 提高概率
        ),
        
        # ✅ 弹性变形（模拟褶皱）
        A.OneOf([
            A.ElasticTransform(
                alpha=100,          # 增加变形强度
                sigma=10,
                p=1.0
            ),
            A.GridDistortion(
                num_steps=8,        # 增加网格密度
                distort_limit=0.5,  # 增加扭曲程度
                p=1.0
            ),
            A.OpticalDistortion(
                distort_limit=0.3,
                shift_limit=0.3,
                p=1.0
            ),
        ], p=0.7),  # 提高概率
        
        # 运动模糊（模拟飘动）
        A.MotionBlur(blur_limit=(3, 15), p=0.4),  # 提高强度
        
        # 高斯模糊（模拟失焦）
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ])
    
    # ✅ 光照和颜色增强
    color_aug = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.6
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.CLAHE(clip_limit=4.0, p=0.3),  # 自适应直方图均衡
    ])
    
    augmented = [img_pil]
    
    for _ in range(num_aug):
        # 空间变换
        aug_np = spatial_aug(image=img_np)['image']
        
        # 颜色变换
        aug_np = color_aug(image=aug_np)['image']
        
        aug_pil = Image.fromarray(aug_np)
        
        # 额外的PIL增强
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(aug_pil)
            aug_pil = enhancer.enhance(random.uniform(0.6, 1.4))
        
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(aug_pil)
            aug_pil = enhancer.enhance(random.uniform(0.7, 1.3))
        
        if random.random() > 0.3:
            enhancer = ImageEnhance.Color(aug_pil)
            aug_pil = enhancer.enhance(random.uniform(0.7, 1.3))
        
        # ✅ 添加噪声（模拟拍摄噪声）
        if random.random() > 0.5:
            aug_np = np.array(aug_pil)
            noise = np.random.normal(0, 10, aug_np.shape).astype(np.uint8)
            aug_np = np.clip(aug_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            aug_pil = Image.fromarray(aug_np)
        
        augmented.append(aug_pil)
    
    return augmented


# ==================== 编码函数 ====================

def face_encode_enhanced(face_dir_path, output_path='data/faces_enhanced.pkl', 
                        num_aug=30, ensure_face_visible=True):
    '''增强版人脸编码（针对侧脸优化）'''
    if not os.path.exists(face_dir_path):
        print(f"❌ 目录不存在: {face_dir_path}")
        return None
    
    encode_dict = {}
    failed_images = []
    
    print("=" * 60)
    print(f"开始编码人脸数据 (针对侧脸优化)...")
    print(f"增强数量: {num_aug}, 确保人脸可见: {ensure_face_visible}")
    print("=" * 60)
    
    file_list = [f for f in os.listdir(face_dir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for idx, path in enumerate(file_list):
        file_path = os.path.join(face_dir_path, path)
        name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            img_pil = preprocess_image(file_path)
            
            if min(img_pil.size) < 100:
                print(f"  [{idx+1}/{len(file_list)}] ⚠️  {name}: 图像太小 {img_pil.size}")
            
            # 先检查原图
            test_array = np.array(img_pil)
            face_locations = face_recognition.face_locations(test_array)
            
            # ✅ 使用large模型检测（对侧脸更好）
            try:
                test_enc = face_recognition.face_encodings(test_array, face_locations, model='large')
            except:
                test_enc = face_recognition.face_encodings(test_array, face_locations)
            
            if len(test_enc) == 0:
                failed_images.append(name)
                print(f"  [{idx+1}/{len(file_list)}] ❌ {name}: 原图未检测到人脸")
                continue
            
            if len(face_locations) > 1:
                print(f"  [{idx+1}/{len(file_list)}] ⚠️  {name}: 检测到 {len(face_locations)} 张人脸，使用第一张")
            
            # 生成增强图像
            augmented_imgs = augment_face_image_robust(img_pil, num_aug, ensure_face_visible)
            
            # 编码
            encodings = []
            for aug_img in augmented_imgs:
                img_array = np.array(aug_img)
                # ✅ 使用large模型编码
                try:
                    enc = face_recognition.face_encodings(img_array, model='large')
                except:
                    enc = face_recognition.face_encodings(img_array)
                
                if len(enc) > 0:
                    encodings.append(enc[0])
            
            if len(encodings) > 0:
                encode_dict[name] = np.array(encodings)
                expected = num_aug + 1
                status = "✅" if len(encodings) >= expected * 0.8 else "⚠️"
                print(f"  [{idx+1}/{len(file_list)}] {status} {name}: {len(encodings)}/{expected} 个编码")
            else:
                failed_images.append(name)
                print(f"  [{idx+1}/{len(file_list)}] ❌ {name}: 所有增强版本都未检测到人脸")
                
        except Exception as e:
            failed_images.append(name)
            print(f"  [{idx+1}/{len(file_list)}] ❌ {name}: {e}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(encode_dict, output_path)
    
    print("=" * 60)
    print(f"✅ 编码完成！")
    print(f"   成功: {len(encode_dict)} 人")
    print(f"   失败: {len(failed_images)} 人")
    if failed_images:
        print(f"   失败列表: {failed_images}")
    print(f"   保存路径: {output_path}")
    print("=" * 60)
    
    return encode_dict


def encode_flags_enhanced(flag_dir_path='flags', output_path='data/flags_enhanced.pkl', 
                         num_aug=40, model_path='model/cn-clip/'):
    '''增强版旗帜编码（针对褶皱倾斜优化）'''
    if not os.path.exists(flag_dir_path):
        print(f"❌ 目录不存在: {flag_dir_path}")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    model, preprocess = load_from_name('ViT-B-16', device=device, download_root=model_path)
    model.eval()
    
    dic = {}
    print("=" * 60)
    print(f"开始编码旗帜数据 (针对褶皱倾斜优化)...")
    print(f"增强数量: {num_aug}")
    print("=" * 60)
    
    file_list = [f for f in os.listdir(flag_dir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for idx, filename in enumerate(file_list):
        img_path = os.path.join(flag_dir_path, filename)
        name = os.path.splitext(filename)[0]
        
        try:
            img_pil = preprocess_image(img_path)
            
            # 生成增强图像
            augmented_imgs = augment_flag_image_robust(img_pil, num_aug)
            
            # 编码
            encodings = []
            for aug_img in augmented_imgs:
                inputs = preprocess(aug_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model.encode_image(inputs)
                    feat /= feat.norm(dim=-1, keepdim=True)
                    encodings.append(feat.squeeze(0).cpu())
            
            dic[name] = torch.stack(encodings)
            expected = num_aug + 1
            status = "✅" if len(encodings) == expected else "⚠️"
            print(f"  [{idx+1}/{len(file_list)}] {status} {name}: {len(encodings)}/{expected} 个编码")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(file_list)}] ❌ {name}: {e}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dic, output_path)
    
    print("=" * 60)
    print(f"✅ 旗帜编码完成！共 {len(dic)} 个")
    print(f"   保存路径: {output_path}")
    print("=" * 60)
    
    return dic


# ==================== 主函数 ====================

if __name__ == '__main__':
    print("\n🚀 开始鲁棒性增强编码...\n")
    
    # 人脸编码 - 针对侧脸优化
    face_encode_enhanced(
        face_dir_path='faces',
        output_path='data/faces_robust.pkl',  # 新文件名
        num_aug=30,  # 增加到30
        ensure_face_visible=True
    )
    
    print("\n" + "="*60 + "\n")
    
    # 旗帜编码 - 针对褶皱倾斜优化
    encode_flags_enhanced(
        flag_dir_path='flags',
        output_path='data/flags_robust.pkl',  # 新文件名
        num_aug=40,  # 增加到40
        model_path='model/cn-clip/'
    )
    
    print("\n🎉 鲁棒性增强编码完成！\n")