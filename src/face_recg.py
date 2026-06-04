import face_recognition
import os
import torch
import numpy as np


def face_encode(face_dir_path):
    '''
    编码人脸特征
    '''
    encode_lst = []  # 人脸特征向量
    name_lst = []  # 人名
    # 遍历目录中所有文件
    for path in os.listdir(face_dir_path):
        file_path = os.path.join(face_dir_path,path)
        encode_lst.append(face_recognition.face_encodings(face_recognition.load_image_file(file_path))[0])  # 编码
        name_lst.append(os.path.basename(file_path)[:-4])  # 提取人名
    torch.save((encode_lst, name_lst), 'data/faces_v6.pkl')  # 保存编码文件


# def face_recg(face_path, known_face='data/faces_v6.pkl'):
#     '''
#     人脸识别函数
#     '''
#     # 加载已知人脸数据
#     known_face_encoding, known_name = torch.load(known_face)  
#     # 提取待识别人脸特征
#     ukn_encoding = face_recognition.face_encodings(face_recognition.load_image_file(face_path))
#     if len(ukn_encoding) != 1:  # 若不是一张人脸
#         return 'Unknown'
#     else:
#         ukn_encoding = ukn_encoding[0]  # 获取唯一人脸编码
#     face_distances = face_recognition.face_distance(known_face_encoding, ukn_encoding)  # 计算与所有已知人脸的距离
#     best_index = np.argmin(face_distances)  # 找出最相似的人脸
#     # 判断是否真正匹配
#     if face_recognition.compare_faces([known_face_encoding[best_index]], ukn_encoding)[0]:
#         name = known_name[best_index]
#     else:
#         name = 'Unknown'
#     return name


def face_recg(face_path, known_face='data/faces_v6.pkl', threshold=0.55, min_gap=0.02):
    '''
    人脸识别函数
    '''
    # 加载已知人脸数据
    known_face_encoding, known_name = torch.load(known_face)  
    # 提取待识别人脸特征
    ukn_encoding = face_recognition.face_encodings(face_recognition.load_image_file(face_path))
    if len(ukn_encoding) != 1:  # 若不是一张人脸
        return 'Unknown'
    else:
        ukn_encoding = ukn_encoding[0]  # 获取唯一人脸编码
    face_distances = face_recognition.face_distance(known_face_encoding, ukn_encoding)  # 计算与所有已知人脸的距离
    
    # 找出最相似的前两名
    sorted_indices = np.argsort(face_distances)
    best_index = sorted_indices[0]
    best_distance = face_distances[best_index]

    # 距离阈值
    if best_distance > threshold:
        return 'Unknown'

    # 区分度检查
    if len(sorted_indices) > 1:
        second_distance = face_distances[sorted_indices[1]]
        gap = second_distance - best_distance
        
        if gap < min_gap:  # 与第二名太接近，拒绝
            return 'Unknown'

    return known_name[best_index]


# def face_recg(face_path, known_face='data/faces_v6.pkl', threshold=0.55, min_gap=0.02, 
#               verbose=False, return_detail=False):
#     '''
#     人脸识别函数 - 带动态Gap策略
    
#     参数:
#         verbose: 是否打印详细日志
#         return_detail: 是否返回详细信息（字典格式）
#     '''
#     # 加载已知人脸数据
#     known_face_encoding, known_name = torch.load(known_face)  
    
#     # 提取待识别人脸特征
#     ukn_encoding = face_recognition.face_encodings(face_recognition.load_image_file(face_path))
    
#     if len(ukn_encoding) != 1:  # 若不是一张人脸
#         if verbose:
#             print(f"  [FACE] 检测到 {len(ukn_encoding)} 个人脸（需要恰好1个）")
#         if return_detail:
#             return {
#                 'name': 'Unknown',
#                 'reason': f'detected_{len(ukn_encoding)}_faces',
#                 'score': None,
#                 'gap': None
#             }
#         return 'Unknown'
#     else:
#         ukn_encoding = ukn_encoding[0]
    
#     # 计算与所有已知人脸的距离
#     face_distances = face_recognition.face_distance(known_face_encoding, ukn_encoding)
    
#     # 找出最相似的前几名
#     sorted_indices = np.argsort(face_distances)
#     best_index = sorted_indices[0]
#     best_distance = face_distances[best_index]
#     best_name = known_name[best_index]
    
#     # 收集Top3信息（用于调试）
#     top3 = []
#     for i in range(min(3, len(sorted_indices))):
#         idx = sorted_indices[i]
#         top3.append({
#             'name': known_name[idx],
#             'distance': face_distances[idx]
#         })
    
#     if verbose:
#         print(f"  [FACE] Top3匹配结果:")
#         for i, item in enumerate(top3, 1):
#             marker = "✓" if i == 1 else " "
#             print(f"    {marker} #{i}: {item['name']:20s} 距离={item['distance']:.4f}")
    
#     # 计算区分度
#     gap = None
#     second_name = None
#     second_distance = None
#     if len(sorted_indices) > 1:
#         second_index = sorted_indices[1]
#         second_distance = face_distances[second_index]
#         second_name = known_name[second_index]
#         gap = second_distance - best_distance
        
#         if verbose:
#             print(f"  [FACE] 区分度 Gap = {gap:.4f} (第二名: {second_name})")
    
#     # 阈值检查
#     passed_threshold = best_distance <= threshold
    
#     if verbose:
#         print(f"  [FACE] 距离检查: {best_distance:.4f} <= {threshold}? {passed_threshold}")
    
#     # ========== 动态 Gap 策略 ==========
#     if passed_threshold:
#         # 根据距离动态调整gap阈值
#         if best_distance < 0.40:         # 非常确定（距离很小）
#             effective_gap = min_gap * 0.4
#             confidence_level = "非常确定"
#         elif best_distance < 0.44:       # 比较确定
#             effective_gap = min_gap * 0.6
#             confidence_level = "比较确定"
#         elif best_distance < 0.47:       # 一般确定
#             effective_gap = min_gap * 0.8
#             confidence_level = "一般确定"
#         else:                            # 勉强通过
#             effective_gap = min_gap * 1.0
#             confidence_level = "勉强通过"
        
#         if verbose:
#             print(f"  [FACE] 置信度: {confidence_level} (距离={best_distance:.4f})")
#             print(f"  [FACE] 动态Gap阈值: {effective_gap:.4f} (原始={min_gap:.4f}, 系数={effective_gap/min_gap:.2f})")
#     else:
#         # 如果连距离阈值都没通过，gap就无所谓了
#         effective_gap = min_gap
    
#     passed_gap = (gap is None) or (gap >= effective_gap)
    
#     if verbose and gap is not None:
#         print(f"  [FACE] Gap检查: {gap:.4f} >= {effective_gap:.4f}? {passed_gap}")
    
#     # 判定结果
#     if not passed_threshold:
#         reason = 'distance_too_large'
#         result_name = 'Unknown'
#         if verbose:
#             print(f"  [FACE] ✗ 距离过大 ({best_distance:.4f} > {threshold})")
#     elif not passed_gap:
#         reason = 'gap_too_small'
#         result_name = 'Unknown'
#         if verbose:
#             print(f"  [FACE] ✗ 区分度不足 ({gap:.4f} < {effective_gap:.4f})")
#     else:
#         reason = 'success'
#         result_name = best_name
#         if verbose:
#             print(f"  [FACE] ✓ 识别成功: {best_name}")
    
#     # 返回详细信息或仅返回名字
#     if return_detail:
#         return {
#             'name': result_name,
#             'reason': reason,
#             'score': best_distance,  # 注意：距离越小越好
#             'gap': gap,
#             'threshold': threshold,
#             'min_gap': min_gap,
#             'effective_gap': effective_gap if passed_threshold else None,  # 新增：实际使用的gap阈值
#             'confidence_level': confidence_level if passed_threshold else None,  # 新增：置信度等级
#             'passed_threshold': passed_threshold,
#             'passed_gap': passed_gap,
#             'top3': top3,
#             'second_name': second_name,
#             'second_score': second_distance
#         }
    
#     return result_name