#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import os
import pymysql

# =========================================================
# 数据库配置
# =========================================================
MYSQL_HOST = "117.50.220.157"
MYSQL_USER = "root"
MYSQL_PORT = 3306
MYSQL_PWD = "qqzcy@1014"
MYSQL_DB = "spider_info"

# 结构化标签的前缀及其期望的顺序
STRUCTURED_PREFIX_ORDER = ['是否涉华-', '情感倾向-', '涉华情况分类-']

# =========================================================
# 原始词表
# =========================================================
person_list = [
    '丘吉尔', '习近平', '亚努科维奇', '亨特拜登', '伊丽莎白二世',
    '伊塔马尔·本-格维尔（以色列国家安全部部长）', '伯恩斯', '佩斯科夫（俄罗斯总统办公厅副主任-总统新闻秘书）',
    '佩洛西', '侯友宜', '克林顿', '内塔尼亚胡', '冯德莱恩', '凯文麦卡锡', '利兹切尼',
    '华春莹', '卡德罗夫', '卢卡申科（白俄总统）', '叶利钦', '吉尔拜登', '哈梅内伊',
    '哈里斯', '埃尔多安', '基辛格', '奥巴马', '奥斯汀', '威廉·约瑟夫·伯恩斯（中情局局长）',
    '孙中山', '安倍晋三', '宋楚瑜', '小布什', '尤利娅·季莫申科（乌克兰前总理）', '尹锡悦',
    '尼基黑利', '尼科尔·帕希尼扬（亚美尼亚总理）', '岸田文雄', '崔天凯', '巴沙尔阿萨德',
    '布林肯', '布热津斯基', '希拉里', '希特勒', '帕特鲁舍夫·尼古拉·波拉丹诺维奇（俄安全会议秘书）',
    '彭斯', '德桑蒂斯', '扎卢日内', '托卡耶夫', '拉夫罗夫', '拜登', '文在寅', '斯大林',
    '普京', '普里戈任', '朔尔茨', '朴槿惠', '李家超', '李登辉', '林郑月娥', '柯文哲',
    '梅德韦杰夫', '梅拉尼娅特朗普', '梅洛尼', '欧尔班（匈牙利总理）', '武契奇',
    '江启臣（台湾“立法院副院长”）', '汪文斌', '沃洛金（俄罗斯联邦国家杜马主席）',
    '波罗申科波罗申科（乌克兰前总统）', '泽连斯基', '特朗普', '特鲁多', '王毅', '玛丽勒庞',
    '穆罕默德巴祖姆', '米洛舍维奇（塞尔维亚前总理）', '米舒斯京（俄罗斯总理）', '纽特·金里奇',
    '绍伊古', '罗斯福', '老布什', '耿爽', '肯尼迪', '苏嘉全（蔡英文办公室原秘书长）',
    '苏纳克', '苏莱曼尼', '苏贞昌（台当局前“行政院长”）', '莫迪', '菅义伟', '萨科齐',
    '萨达姆', '蒋万安', '蒋介石', '蓬佩奥', '蔡英文', '谢尔盖·里亚布科夫（俄副外交部长）',
    '赖清德', '赵立坚', '连战', '郝龙斌', '里根', '金正恩', '阿列克谢·纳瓦利内（曾任俄罗斯顾问）',
    '陈其迈', '陈时中', '陈水扁', '韩国瑜', '马克米利', '马克龙', '马杜罗（委内瑞拉总统）',
    '马泰乌什·莫拉维茨基（波兰前总理）', '马英九', '鲁斯特姆·乌梅罗夫（乌克兰国防部长）',
    '鲍里斯约翰逊', '黄之锋'
]

countries = [
    '斯洛伐克', '南非', '伊朗', '冰岛', '拉脱维亚', '葡萄牙', '约旦', '贝宁',
    '吉尔吉斯斯坦', '利比果亚', '莱索托', '哈萨克斯坦', '东帝汶', '刚果(金)', '斯威士兰',
    '安提瓜和巴布达', '巴林', '阿富汗', '阿尔及利亚', '博茨瓦纳', '加蓬', '马里',
    '巴巴多斯', '几内亚比绍', '塞拉利昂', '阿拉伯国家联盟', '瑙鲁', '加拿大', '美国国徽',
    '圣多美和普林西比', '格林纳达', '爱沙尼亚', '莫桑比克', '保加利亚', '梵蒂冈', '苏里南',
    '缅甸', '布隆迪', '比利时', '奥地利', '多米尼克', '肯尼亚', '欧洲联盟', '圣卢西亚',
    '以色列', '尼加拉瓜', '安哥拉', '卢旺达', '挪威', '马来西亚', '纳米比亚', '美国',
    '洪都拉斯', '智利', '哥伦比亚', '意大利', '克罗地亚', '吉布提', '土库曼斯坦', '卡塔尔',
    '印度尼西亚', '厄立特里亚', '瑞典', '多哥', '特立尼达和多巴哥', '坦桑尼亚', '加纳',
    '基里巴斯', '芬兰', '巴基斯坦', '越南', '塔吉克斯坦', '毛里塔尼亚', '圣文森特和格林纳丁斯',
    '爱尔兰', '图瓦卢', '汤加', '哥斯达黎加', '不丹', '亚美尼亚', '斯洛文尼亚', '蒙古',
    '秘鲁', '西萨摩亚', '巴勒斯坦', '圣马力诺', '津巴布韦', '利比亚', '塞尔维亚', '多米尼加',
    '乌克兰', '德国', '朝鲜', '古巴', '斐济', '阿曼', '罗马尼亚', '帕劳', '佛得角', '菲律宾',
    '斯里兰卡', '冈比亚', '白俄罗斯', '格鲁吉亚', '尼日利亚', '匈牙利', '印度', '苏丹',
    '萨尔瓦多', '俄罗斯', '塞内加尔', '马绍尔群岛', '日本', '巴西', '厄瓜多尔', '阿塞拜疆',
    '阿根廷', '法国', '阿尔巴尼亚', '丹麦', '埃及', '澳大利亚', '乍得', '喀麦隆',
    '列支敦士登', '柬埔寨', '孟加拉国', '委内瑞拉', '土耳其', '赞比亚', '巴拉圭', '突尼斯',
    '巴哈马', '赤道几内亚', '科威特', '尼日尔', '中非', '摩洛哥', '马耳他', '巴布亚新几内亚',
    '乌兹别克斯坦', '韩国', '希腊', '英国', '捷克', '瑞士', '乌干达', '危地马拉', '玻利维亚',
    '所罗门群岛', '卢森堡', '沙特阿拉伯', '马拉维', '圣基茨和尼维斯', '塞浦路斯', '索马里',
    '科摩罗', '立陶宛', '几内亚', '新西兰', '马尔代夫', '布基纳法索', '海地', '黎巴嫩',
    '马达加斯加', '刚果', '圭亚那', '中国', '马其顿', '摩纳哥', '波斯尼亚和黑塞哥维那',
    '埃塞俄比亚', '塞舌尔', '摩尔多瓦', '牙买加', '尼泊尔', '西班牙', '伊拉克', '也门',
    '墨西哥', '叙利亚', '乌拉圭', '老挝', '泰国', '科特迪瓦', '安道尔', '波兰', '文莱',
    '荷兰', '毛里求斯', '瓦努阿图', '巴拿马', '阿联酋', '新加坡'
]

# =========================================================
# 文本清洗 & 别名映射构建
# =========================================================
def clean_name(name):
    """去掉括号内容：习近平（国家主席） → 习近平"""
    return re.sub(r'（.*?）|\(.*?\)', '', name).strip()

def fix_duplicate(name):
    """修复重复拼接错误：波罗申科波罗申科 → 波罗申科"""
    if len(name) % 2 == 0:
        half = name[:len(name)//2]
        if half * 2 == name:
            return half
    return name

def build_person_alias(person_list):
    """
    构建别名映射（标准名 → 标准名）
    短别名仅当未被占用时添加，避免冲突。
    """
    alias_dict = {}
    # 先记录所有标准名
    for full_name in person_list:
        std = fix_duplicate(clean_name(full_name))
        if std:
            alias_dict[std] = std

    # 添加短别名（仅当尚未映射）
    for full_name in person_list:
        std = fix_duplicate(clean_name(full_name))
        if not std:
            continue

        if '·' in std:
            parts = std.split('·')
            last = parts[-1]
            if len(last) >= 2 and last not in alias_dict:
                alias_dict[last] = std
            if '-' in last:
                for sub in last.split('-'):
                    if len(sub) >= 2 and sub not in alias_dict:
                        alias_dict[sub] = std
        elif '-' in std:
            for sub in std.split('-'):
                if len(sub) >= 2 and sub not in alias_dict:
                    alias_dict[sub] = std

    return alias_dict

# 构建国家集合（小写，去空格）
country_set_lower = {c.strip().lower() for c in countries if c.strip()}

def match_person(name, alias_dict):
    """人名匹配"""
    name_clean = name.strip()
    return alias_dict.get(name_clean, None)

def match_flag(name):
    """国家匹配（忽略大小写）"""
    name_clean = name.strip().lower()
    return name_clean if name_clean in country_set_lower else None

# =========================================================
# 标签拆分与顺序重组
# =========================================================
def split_tags(tag_str):
    """
    将旧tag拆成两部分：
    - structured: 以 STRUCTURED_PREFIX_ORDER 中任一前缀开头的标签
    - entities:   其他所有标签（人名、国家等实体）
    """
    structured = set()
    entities = set()
    if not tag_str:
        return structured, entities
    for tag in tag_str.split(','):
        tag = tag.strip()
        if not tag:
            continue
        # 判断是否为结构化标签（以前缀开头）
        is_structured = any(tag.startswith(prefix) for prefix in STRUCTURED_PREFIX_ORDER)
        if is_structured:
            structured.add(tag)
        else:
            entities.add(tag)
    return structured, entities

def order_structured_tags(structured_set):
    """
    按照 STRUCTURED_PREFIX_ORDER 的顺序返回结构化标签列表。
    每个前缀最多出现一次（实际存储中通常只有一个）。
    """
    ordered = []
    for prefix in STRUCTURED_PREFIX_ORDER:
        # 找出以该前缀开头的标签（只有一个，但用循环取第一个）
        for tag in structured_set:
            if tag.startswith(prefix):
                ordered.append(tag)
                break   # 每个前缀只取一个
    return ordered

# =========================================================
# 数据库操作白名单
# =========================================================
ALLOWED_TABLES = {'collect_content', 'anchor_zb_video'}
ALLOWED_ID_COLS = {'content_id', 'id'}
ALLOWED_TAG_COLS = {'analysis_tag'}

def validate_db_params(table_name, id_col, tag_col):
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"非法表名: {table_name}")
    if id_col not in ALLOWED_ID_COLS:
        raise ValueError(f"非法ID列: {id_col}")
    if tag_col not in ALLOWED_TAG_COLS:
        raise ValueError(f"非法标签列: {tag_col}")

# =========================================================
# 核心处理函数
# =========================================================
def process_json(json_path, table_name, id_col, tag_col, alias_dict):
    """
    主流程：JSON → 实体抽取 → 合并旧tag → 按顺序重组 → 更新数据库
    """
    validate_db_params(table_name, id_col, tag_col)

    if not os.path.exists(json_path):
        print(f"⚠️ 文件不存在，跳过: {json_path}")
        return

    print(f"\n🚀 开始处理 {json_path} → {table_name}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conn = None
    cursor = None
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            port=MYSQL_PORT,
            password=MYSQL_PWD,
            database=MYSQL_DB,
            charset='utf8mb4',
            local_infile=True
        )
        cursor = conn.cursor()

        for video_id, video_info in data.items():
            try:
                video_id = int(video_id)

                detected = set()
                segments = video_info.get("segments", [])

                for seg in segments:
                    text_elem = seg.get("text_elements", {})
                    elements_names = text_elem.get("elements_names", {})
                    per_list = elements_names.get("PER", [])
                    loc_list = elements_names.get("LOC", [])

                    for name in per_list:
                        if len(name) < 2:
                            continue
                        matched = match_person(name, alias_dict)
                        if matched:
                            detected.add(matched)

                    for name in loc_list:
                        matched = match_flag(name)
                        if matched:
                            # 存储原始国家名称（保持原样，不转小写）
                            detected.add(name.strip())

                if not detected:
                    continue

                # 查询旧标签
                cursor.execute(
                    f"SELECT {tag_col} FROM {table_name} WHERE {id_col}=%s",
                    (video_id,)
                )
                row = cursor.fetchone()
                if not row:
                    print(f"⚠️ id={video_id} 在 {table_name} 中不存在，跳过")
                    continue

                old_tags = row[0] or ''
                structured_set, old_entities = split_tags(old_tags)

                # 合并实体（保留集合去重）
                new_entities = old_entities | detected

                # 按顺序重组标签
                ordered_structured = order_structured_tags(structured_set)
                # 实体部分：不排序也可，但为了一致性可排序（不影响顺序要求）
                sorted_entities = sorted(new_entities)   # 按字符串排序
                final_tags = ordered_structured + sorted_entities
                new_tags = ','.join(final_tags)

                # 更新数据库
                cursor.execute(
                    f"UPDATE {table_name} SET {tag_col}=%s WHERE {id_col}=%s",
                    (new_tags, video_id)
                )
                print(f"✅ id={video_id} 更新完成: {new_tags}")

            except Exception as e:
                print(f"❌ 处理 id={video_id} 时出错: {e}")
                continue

        conn.commit()
        print(f"🎉 完成 {table_name}")

    except Exception as e:
        print(f"❌ 数据库操作失败: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# =========================================================
# 主入口
# =========================================================
if __name__ == "__main__":
    person_alias = build_person_alias(person_list)

    # 处理视频报告（collect_content表）
    process_json(
        json_path="/data/srch_vt/load_tmp/video_report.json",
        table_name="collect_content",
        id_col="content_id",
        tag_col="analysis_tag",
        alias_dict=person_alias
    )

    # 处理直播报告（anchor_zb_video表）
    process_json(
        json_path="/data/srch_vt/load_tmp/zb_video_report.json",
        table_name="anchor_zb_video",
        id_col="id",
        tag_col="analysis_tag",
        alias_dict=person_alias
    )