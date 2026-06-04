import os
import cv2
import torch
import shutil
from PIL import Image
from face_recg import face_recg
from my_utils import rgba2rgb, search_subimg


def img_recg(processor, class_model, CLIP_MODEL, pic,
             face_threshold=0.55, face_gap=0.04, 
             flag_threshold=0.65, flag_gap=0.04,
             verbose=False):
    
    element_dict = {
        'img': '',
        'elements': [],
        'class': 0
    }

    img = Image.open(pic)
    
    # 修改这里：只对RGBA图片进行转换
    if img.mode == 'RGBA':
        img = rgba2rgb(img)
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    inputs = processor(images=img, return_tensors="pt").to('cuda:0')
    pred = class_model(inputs.data['pixel_values']).to('cuda:0')
    pred_label = torch.argmax(pred).item()
    
    if verbose:
        class_names = ['人脸', '国旗', '武器装备', '杂物']
        print(f"    [分类器] 预测类别: {class_names[pred_label]} (label={pred_label})")
    
    key_words = []
    
    if pred_label == 0:
        element_dict['img'] = os.path.join(os.getcwd(), pic)
        
        if verbose:
            print(f"    [人脸识别] 开始识别...")
        
        # 使用调试版本的 face_recg
        person_result = face_recg(pic, threshold=face_threshold, min_gap=face_gap,
                                 verbose=verbose, return_detail=True)
        
        person_name = person_result['name'] if isinstance(person_result, dict) else person_result
        
        if verbose and isinstance(person_result, dict):
            print(f"    [人脸识别] 最终结果: {person_name}")
            if person_name == 'Unknown':
                print(f"    [人脸识别] ✗ 失败原因: {person_result.get('reason', 'unknown')}")
                if person_result.get('score') is not None:
                    print(f"    [人脸识别]   - 最佳距离: {person_result['score']:.4f}, 阈值: {face_threshold}")
                if person_result.get('gap') is not None:
                    eff_gap = person_result.get('effective_gap', face_gap)
                    print(f"    [人脸识别]   - 实际Gap: {person_result['gap']:.4f}, 需要Gap: {eff_gap:.4f}")
                if person_result.get('confidence_level'):
                    print(f"    [人脸识别]   - 置信度等级: {person_result['confidence_level']}")
            else:
                print(f"    [人脸识别] ✓ 成功: {person_name}")
                if person_result.get('confidence_level'):
                    print(f"    [人脸识别]   - 置信度等级: {person_result['confidence_level']}")
        
        element_dict['elements'] = [person_name]
        # 添加详细信息到 element_dict（用于后续统计）
        element_dict['detail'] = person_result if isinstance(person_result, dict) else None
        
        if person_name != 'Unknown':
            key_words.append(person_name)
    
    elif pred_label == 1:
        element_dict['img'] = os.path.join(os.getcwd(), pic)
        
        if verbose:
            print(f"    [旗帜识别] 开始识别...")
        
        # 使用调试版本的 search_subimg
        flag_result = search_subimg(img, 'flag', 
                                   CLIP_MODEL.get_model(), 
                                   CLIP_MODEL.get_preprocess(),
                                   threshold=flag_threshold, 
                                   min_gap=flag_gap,
                                   verbose=verbose,
                                   return_detail=True)
        
        org_name = flag_result['name'] if isinstance(flag_result, dict) else flag_result
        
        if verbose and isinstance(flag_result, dict):
            print(f"    [旗帜识别] 最终结果: {org_name}")
            if org_name == 'Unknown':
                print(f"    [旗帜识别] ✗ 失败原因: {flag_result.get('reason', 'unknown')}")
                if flag_result.get('score') is not None:
                    print(f"    [旗帜识别]   - 最佳分数: {flag_result['score']:.4f}, 阈值: {flag_threshold}")
                if flag_result.get('gap') is not None:
                    eff_gap = flag_result.get('effective_gap', flag_gap)
                    print(f"    [旗帜识别]   - 实际Gap: {flag_result['gap']:.4f}, 需要Gap: {eff_gap:.4f}")
                if flag_result.get('confidence_level'):
                    print(f"    [旗帜识别]   - 置信度等级: {flag_result['confidence_level']}")
            else:
                print(f"    [旗帜识别] ✓ 成功: {org_name}")
                if flag_result.get('confidence_level'):
                    print(f"    [旗帜识别]   - 置信度等级: {flag_result['confidence_level']}")
        
        element_dict['elements'] = [org_name]
        element_dict['class'] = 1
        # 添加详细信息到 element_dict
        element_dict['detail'] = flag_result if isinstance(flag_result, dict) else None
        
        if org_name != 'Unknown':
            key_words.append(org_name)
    
    elif pred_label == 2:
        element_dict['img'] = os.path.join(os.getcwd(), pic)
        element_dict['elements'] = ['武器装备']
        element_dict['class'] = 2
        
        if verbose:
            print(f"    [武器装备] 识别为武器装备")
    
    else:
        element_dict['elements'] = ['杂物']
        element_dict['img'] = os.path.join(os.getcwd(), pic)
        element_dict['class'] = 3
        
        if verbose:
            print(f"    [杂物] 识别为杂物")
    
    return element_dict, key_words


def get_imgs(frame_dir):
    """获取目录下的所有图片，按文件名排序"""
    imgs = []
    if not os.path.exists(frame_dir):
        return imgs, []
    
    files = []
    for f in os.listdir(frame_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            files.append(f)
    
    # 按文件名排序，确保顺序一致
    files.sort()
    
    for f in files:
        imgs.append(os.path.join(frame_dir, f))
    
    return imgs, imgs


if __name__ == "__main__":
    # 初始化模型
    from transformers import ViTImageProcessor
    from operations.graph_search import SearchGraph
    from operations.keywords_search import TFIDFSearch
    from encode import ChineseCLIP
    from segment_helpers import Segment_generator, get_segment_images
    from vit_model import vit_model
    from config import TFIDF_PATH, CLASS_CHECKPOINT, CLASS_MODEL_PATH

    with torch.no_grad():
        CLIP_MODEL = ChineseCLIP()
        SAM = Segment_generator()
        SAM = SAM.get_model()
        tfidf_search = TFIDFSearch(None, False, TFIDF_PATH)
        search_graph = SearchGraph(TFIDF_PATH)
        processor = ViTImageProcessor.from_pretrained(CLASS_MODEL_PATH)
        class_model = vit_model().to('cuda:0')
        class_model.load_state_dict(torch.load(CLASS_CHECKPOINT))
    
    # ==================== 阈值配置区 ====================
    # 方案A：保守型（推荐先试这个）
    face_threshold = 0.48
    face_gap = 0.045
    flag_threshold = 0.87
    flag_gap = 0.05
    
    # 方案B：激进型（如果方案A漏检太多，可以试这个）
    # face_threshold = 0.50
    # face_gap = 0.04
    # flag_threshold = 0.85
    # flag_gap = 0.04
    
    # 是否启用详细日志
    verbose_mode = True  # 设为 False 可以关闭详细日志
    # ===================================================
    
    # 输入输出目录
    input_frames_dir = "/data/srch_vt/tmp/debug_frames"
    segment_base_dir = "tmp/segment_frames"
    
    # 清理并创建输出目录
    if os.path.exists(segment_base_dir):
        shutil.rmtree(segment_base_dir)
    os.makedirs(segment_base_dir, exist_ok=True)
    
    # 获取所有帧图片
    print("正在读取帧图片...")
    frame_files = []
    for f in os.listdir(input_frames_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_files.append(f)
    
    # 排序确保处理顺序一致
    frame_files.sort()
    
    # ========== 测试配置 ==========
    # 测试单独一个
    # frame_files = ["/data/srch_vt/tmp/debug_frames/frame_1893.jpg"]
    # 或者测试多个（取消注释）
    # frame_files = frame_files[:10]  # 测试前10帧
    # ==============================
    
    print(f"找到 {len(frame_files)} 个帧图片")
    print(f"\n当前阈值设置:")
    print(f"  人脸: threshold={face_threshold}, min_gap={face_gap}")
    print(f"  旗帜: threshold={flag_threshold}, min_gap={flag_gap}")
    print(f"详细日志: {'开启' if verbose_mode else '关闭'}\n")
    
    # 初始化统计变量
    frame_persons = []
    frame_nations = []
    frame_others = []
    element_dicts = []
    key_words = []
    all_results = []
    
    # 统计识别情况（增强版）
    recognition_stats = {
        'total_segments': 0,
        'face_attempts': 0,
        'face_success': 0,
        'face_failures': {},  # 按原因统计
        'face_confidence_levels': {},  # 按置信度统计
        'flag_attempts': 0,
        'flag_success': 0,
        'flag_failures': {},
        'flag_confidence_levels': {},
    }
    
    # 处理每一帧
    total_frames = len(frame_files)
    
    for idx, frame_file in enumerate(frame_files):
        print(f"\n{'='*80}")
        print(f"处理第 {idx+1}/{total_frames} 帧: {frame_file}")
        print(f"{'='*80}")
        
        frame_path = os.path.join(input_frames_dir, frame_file)
        
        # 读取帧
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"  警告: 无法读取图片 {frame_file}")
            continue
        
        # 创建该帧的分割目录
        frame_name = os.path.splitext(frame_file)[0]
        frame_dir = os.path.join(segment_base_dir, frame_name)
        os.makedirs(frame_dir, exist_ok=True)
        
        # 使用SAM进行分割
        try:
            if SAM is not None:
                get_segment_images(SAM, frame, frame_dir)
                print(f"  ✓ SAM分割完成，输出到: {frame_dir}")
        except Exception as e:
            print(f"  ✗ SAM分割失败: {e}")
            continue
        
        # 获取分割后的子图
        seg_pics, _ = get_imgs(frame_dir)
        all_pics = [frame_path] + seg_pics
        
        print(f"  ✓ 共 {len(all_pics)} 张图片待处理（1张原图 + {len(seg_pics)}张分割图）\n")
        
        # 保存该帧的结果
        frame_result = {
            'frame_file': frame_file,
            'frame_index': idx + 1,
            'segments': []
        }
        
        # 对每个图片进行识别
        for pic_idx, pic_path in enumerate(all_pics):
            recognition_stats['total_segments'] += 1
            
            if pic_idx == 0:
                pic_type = "原图"
                seg_filename = frame_file
            else:
                pic_type = f"子图{pic_idx}"
                seg_filename = os.path.basename(pic_path)
            
            print(f"  {'─'*76}")
            print(f"  处理{pic_type} {pic_idx+1}/{len(all_pics)}: {seg_filename}")
            
            try:
                element_dict, keyword = img_recg(
                    processor, class_model, CLIP_MODEL, pic_path,
                    face_threshold=face_threshold, 
                    face_gap=face_gap,
                    flag_threshold=flag_threshold, 
                    flag_gap=flag_gap,
                    verbose=verbose_mode
                )
                
                if not element_dict:
                    print(f"    ⚠ 识别结果为空")
                    continue
                
                elem_class = element_dict.get('class', 3)
                class_names = ['人脸', '旗帜', '武器装备', '杂物']
                elements = element_dict.get('elements', [])
                detail = element_dict.get('detail')
                
                # 统计识别情况（增强版）
                if elem_class == 0:  # 人脸
                    recognition_stats['face_attempts'] += 1
                    if elements[0] != 'Unknown':
                        recognition_stats['face_success'] += 1
                        if detail and 'confidence_level' in detail:
                            level = detail['confidence_level']
                            recognition_stats['face_confidence_levels'][level] = \
                                recognition_stats['face_confidence_levels'].get(level, 0) + 1
                    else:
                        if detail and 'reason' in detail:
                            reason = detail['reason']
                            recognition_stats['face_failures'][reason] = \
                                recognition_stats['face_failures'].get(reason, 0) + 1
                
                elif elem_class == 1:  # 旗帜
                    recognition_stats['flag_attempts'] += 1
                    if elements[0] != 'Unknown':
                        recognition_stats['flag_success'] += 1
                        if detail and 'confidence_level' in detail:
                            level = detail['confidence_level']
                            recognition_stats['flag_confidence_levels'][level] = \
                                recognition_stats['flag_confidence_levels'].get(level, 0) + 1
                    else:
                        if detail and 'reason' in detail:
                            reason = detail['reason']
                            recognition_stats['flag_failures'][reason] = \
                                recognition_stats['flag_failures'].get(reason, 0) + 1
                
                if not verbose_mode:
                    print(f"    → 类别: {class_names[elem_class]}")
                    print(f"    → 识别: {elements}")
                    if keyword:
                        print(f"    → 关键词: {keyword}")
                
                # 添加到帧结果
                frame_result['segments'].append({
                    'segment_file': seg_filename,
                    'segment_index': pic_idx + 1,
                    'segment_path': pic_path,
                    'is_original': (pic_idx == 0),
                    'class': elem_class,
                    'class_name': class_names[elem_class],
                    'elements': elements,
                    'keywords': keyword,
                    'detail': detail  # 保存详细信息
                })
                
                # 统计有效识别结果
                if elem_class != 3:
                    if elem_class == 0:
                        frame_persons.append(elements)
                    elif elem_class == 1:
                        frame_nations.append(elements)
                    elif elem_class == 2:
                        frame_others.append(elements)
                    
                    element_dicts.append(element_dict)
                    if keyword:
                        key_words.extend(keyword)
                
            except Exception as e:
                print(f"    ✗ 识别失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        all_results.append(frame_result)
    
    # ==================== 输出统计报告 ====================
    print("\n" + "="*80)
    print("识别效率统计（增强版）")
    print("="*80)
    
    # 人脸识别统计
    if recognition_stats['face_attempts'] > 0:
        face_rate = recognition_stats['face_success'] / recognition_stats['face_attempts'] * 100
        print(f"\n【人脸识别】")
        print(f"  总尝试: {recognition_stats['face_attempts']}")
        print(f"  成功数: {recognition_stats['face_success']}")
        print(f"  成功率: {face_rate:.1f}%")
        
        if recognition_stats['face_confidence_levels']:
            print(f"\n  成功案例的置信度分布:")
            for level, count in sorted(recognition_stats['face_confidence_levels'].items()):
                print(f"    - {level}: {count} 次")
        
        if recognition_stats['face_failures']:
            print(f"\n  失败原因分布:")
            for reason, count in sorted(recognition_stats['face_failures'].items()):
                reason_map = {
                    'distance_too_large': '距离过大',
                    'gap_too_small': '区分度不足',
                    'detected_0_faces': '未检测到人脸',
                    'detected_2_faces': '检测到多张人脸'
                }
                reason_cn = reason_map.get(reason, reason)
                print(f"    - {reason_cn}: {count} 次")
    
    # 旗帜识别统计
    if recognition_stats['flag_attempts'] > 0:
        flag_rate = recognition_stats['flag_success'] / recognition_stats['flag_attempts'] * 100
        print(f"\n【旗帜识别】")
        print(f"  总尝试: {recognition_stats['flag_attempts']}")
        print(f"  成功数: {recognition_stats['flag_success']}")
        print(f"  成功率: {flag_rate:.1f}%")
        
        if recognition_stats['flag_confidence_levels']:
            print(f"\n  成功案例的置信度分布:")
            for level, count in sorted(recognition_stats['flag_confidence_levels'].items()):
                print(f"    - {level}: {count} 次")
        
        if recognition_stats['flag_failures']:
            print(f"\n  失败原因分布:")
            for reason, count in sorted(recognition_stats['flag_failures'].items()):
                reason_map = {
                    'score_too_low': '分数过低',
                    'gap_too_small': '区分度不足',
                    'no_candidates': '无候选特征'
                }
                reason_cn = reason_map.get(reason, reason)
                print(f"    - {reason_cn}: {count} 次")
    
    # 帧与子图对应关系检查
    print("\n" + "="*80)
    print("帧与子图对应关系检查")
    print("="*80)
    
    for frame_result in all_results:
        print(f"\n原始帧: {frame_result['frame_file']} (第{frame_result['frame_index']}帧)")
        print(f"子图数量: {len(frame_result['segments'])}")
        
        for seg in frame_result['segments']:
            marker = "★" if seg['is_original'] else " "
            print(f"  {marker}[{seg['segment_index']}] {seg['segment_file']}")
            print(f"      类别: {seg['class_name']}")
            print(f"      内容: {seg['elements']}")
            
            # 如果有详细信息，也显示出来
            if seg.get('detail') and seg['detail'].get('confidence_level'):
                print(f"      置信度: {seg['detail']['confidence_level']}")
    
    # 总体统计
    print("\n" + "="*80)
    print("识别统计汇总")
    print("="*80)
    
    class_count = [0, 0, 0, 0]
    for elem_dict in element_dicts:
        class_count[elem_dict.get('class', 3)] += 1
    
    class_names = ['人脸', '旗帜', '武器装备', '杂物']
    total_segments = sum(class_count)
    
    print(f"\n总帧数: {total_frames}")
    print(f"总分割数: {total_segments}")
    if total_frames > 0:
        print(f"平均每帧分割数: {total_segments/total_frames:.1f}\n")
    
    print("各类别统计:")
    for i, count in enumerate(class_count):
        percentage = count/total_segments*100 if total_segments > 0 else 0
        print(f"  {class_names[i]}: {count} 个 ({percentage:.1f}%)")
    
    # 统计识别出的具体对象
    recognized_faces = set()
    recognized_flags = set()
    
    for persons in frame_persons:
        for person in persons:
            if person != 'Unknown':
                recognized_faces.add(person)
    
    for nations in frame_nations:
        for nation in nations:
            if nation != 'Unknown':
                recognized_flags.add(nation)
    
    print(f"\n识别出的人物 ({len(recognized_faces)}): {recognized_faces if recognized_faces else '无'}")
    print(f"识别出的旗帜 ({len(recognized_flags)}): {recognized_flags if recognized_flags else '无'}")
    print(f"识别出的关键词总数: {len(key_words)}")
    
    print(f"\n分割结果保存在: {segment_base_dir}")
    print("\n调试完成!")