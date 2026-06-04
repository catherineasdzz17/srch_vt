import json
import gc
import glob
import os
import pymysql
import requests
import subprocess
import sys
import time
import decimal
import traceback
from datetime import datetime, date
from datetime import time as time_module
from typing import Dict, Any, Optional
from tqdm import tqdm
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DB
from service_manager import TmuxServiceManager


def call_tweet_analyze(base_url: str, payload: Dict[str, Any], timeout_s: float = 20.0) -> Optional[Dict[str, Any]]:
    """Call app2's /tweet_analyze endpoint.

    Args:
        base_url: Service base URL, e.g., "http://127.0.0.1:2333".
        payload: Request JSON body. Must include at least 'author_id' and single-tweet fields
                 (e.g., 'text', optional 'tweet_id', 'location', 'publish_date').
        timeout_s: Request timeout in seconds.

    Returns:
        Parsed JSON response dict on success; None on failure.
    """
    url = f"{base_url}/tweet_analyze"
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"request failed: {exc}")
        return None


def _sanitize_payload(obj: Any) -> Any:
    """Convert non-JSON-serializable objects in payload to serializable format."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_payload(item) for item in obj]
    return obj


def safe_service_call(
    base_url: str,
    payload: Dict[str, Any]
) -> Optional[Any]:
    # Sanitize payload to handle datetime/Decimal objects
    sanitized_payload = _sanitize_payload(payload)
    china_resp = call_tweet_analyze(base_url, sanitized_payload)
    return china_resp


def tweets_category_mapping(category_result: int) -> str:
    mapping = {
        0: "其他涉政",
        1: "Fan Gong",
        2: "Min Yun",
        3: "She Tai",
        4: "She Gang",
        5: "She Jiang",
        6: "She Zang",
        7: "She Ling Dao Ren",
        8: "非涉政相关"
    }
    return mapping[category_result]


def analyze_single_tweet_text(base_url: str, text: str, video_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Analyze a single tweet text using the tweet_analyze service.
    
    Args:
        base_url: Service base URL, e.g., "http://127.0.0.1:2333"
        text: The text content to analyze (can be video query result)
        video_id: Optional content_id for database update
        
    Returns:
        Dictionary containing txt_analysis_json data structure, or None on failure
    """
    if not text or not text.strip():
        return None
    
    # Build payload for tweet_analyze service
    payload = {
        "text": text.strip(),
        "content_id": video_id if video_id else 0,
    }
    
    try:
        china_resp = safe_service_call(base_url, payload)
        if not china_resp:
            return None
        
        # Build txt_analysis_json structure
        txt_analysis_data = []
        
        related_value = china_resp.get("related_to_China")
        txt_analysis_data.append({"key": "是否涉华", "value": "是" if related_value else "否"})
        
        entities = china_resp.get("entity")
        txt_analysis_data.append({"key": "涉华关键词以及其频数", "value": entities})
        
        sentiment = china_resp.get("sentiment")
        txt_analysis_data.append({"key": "情感倾向", "value": "正面" if sentiment == "positive" else "负面"})
        
        category_result = china_resp.get("category_result")       
        txt_analysis_data.append({"key": "推文涉华情况分类", "value": tweets_category_mapping(category_result)})
        
        app_in_china = china_resp.get("app_in_China") or china_resp.get("app_in_china")
        txt_analysis_data.append({"key": "应用在华使用情况", "value": "是" if app_in_china else "否"})
        
        app_tweets_res = china_resp.get("app_tweets_res")
        txt_analysis_data.append({"key": "具体在华使用的应用列表", "value": app_tweets_res})
        
        ip_in_china = china_resp.get("ip_in_China") or china_resp.get("ip_in_china")
        txt_analysis_data.append({"key": "IP在华使用情况", "value": "是" if ip_in_china else "否"})
        
        ip_tweets_res = china_resp.get("ip_tweets_res")
        txt_analysis_data.append({"key": "在华ip列表", "value": ip_tweets_res})
        
        place_in_china = china_resp.get("place_in_China") or china_resp.get("place_in_china")
        txt_analysis_data.append({"key": "在华地点情况", "value": "是" if place_in_china else "否"})
        
        place_tweets_res = china_resp.get("place_tweets_res")
        txt_analysis_data.append({"key": "在华地点列表", "value": place_tweets_res})
        
        return {"txt_analysis_json": txt_analysis_data, "raw_response": china_resp}
        
    except Exception as exc:
        print(f"Error analyzing text: {exc}")
        traceback.print_exc()
        return None


class VideoAnalyzer:
    def __init__(self):
        self.db_config = {
            "host": MYSQL_HOST,
            "user": MYSQL_USER,
            "port": MYSQL_PORT,
            "password": MYSQL_PWD,
            "database": MYSQL_DB
        }
        self.base_local_path = "D:/projectVido"
        self.base_url = "http://10.60.208.164:8099/spiderProject/profile"
        self.download_dir = "/data/srch_vt/load_tmp"
        self.tweet_analyze_base_url = "http://127.0.0.1:2333"  # 推文分析服务地址

        self.video_api_base = "http://127.0.0.1:9090"
        self.video_api_timeout = 600  # 提交任务超时10分钟
        self.check_interval = 15  # 每15秒检查一次状态
        self.max_wait_time = 6 * 3600  # 最多等待6小时
        self.max_consecutive_errors = 5  # 最多连续失败5次

        os.makedirs(self.download_dir, exist_ok=True)


    def _submit_video_task(self, video_path, video_info):
        """
        提交视频处理任务到队列API（支持元数据）
        """
        try:
            # 准备任务元数据
            task_data = {
                "table_name": "collect_content",
                "video_id": str(video_info['content_id']),
                "video_url": video_info['video_url'],
                "keyword": video_info['collect_keyword'],
                "create_time": str(video_info['create_time'])
            }
            
            with open(video_path, 'rb') as f:
                files = {
                    'file': (os.path.basename(video_path), f, 'video/mp4')
                }
                data = {
                    'task_request': json.dumps(task_data)
                }
                
                response = requests.post(
                    f'{self.video_api_base}/video_handler/submit',
                    files=files,
                    data=data,
                    timeout=self.video_api_timeout
                )
            
            if response.status_code != 200:
                error_detail = response.text[:500]
                raise Exception(f"提交任务失败，状态码: {response.status_code}, 详情: {error_detail}")
            
            result = response.json()
            task_id = result.get('task_id')
            
            if not task_id:
                raise Exception("服务未返回任务ID")
            
            # 返回任务ID和队列信息
            return {
                'task_id': task_id,
                'queue_position': result.get('queue_position', 0),
                'estimated_wait_time': result.get('estimated_wait_time', 0),
                'queue_size': result.get('queue_size', 0)
            }
            
        except requests.exceptions.Timeout:
            raise Exception(f"上传视频超时 (>{self.video_api_timeout}秒)")
        
        except requests.exceptions.ConnectionError:
            raise Exception("无法连接到视频分析服务")


    def _wait_for_task_completion(self, task_id):
        """
        等待任务完成并返回结果（队列版本）
        """
        max_checks = self.max_wait_time // self.check_interval
        consecutive_errors = 0
        
        # 创建进度条
        with tqdm(
            total=100,
            desc="    处理进度",
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {postfix}',
            ncols=90
        ) as pbar:
            
            last_progress = 0
            
            for i in range(max_checks):
                time.sleep(self.check_interval)
                
                try:
                    # 查询任务状态
                    response = requests.get(
                        f'{self.video_api_base}/video_handler/status/{task_id}',
                        timeout=30
                    )
                    
                    if response.status_code == 404:
                        raise Exception("任务不存在")
                    
                    if response.status_code != 200:
                        consecutive_errors += 1
                        if consecutive_errors >= self.max_consecutive_errors:
                            raise Exception(f"连续{self.max_consecutive_errors}次查询失败")
                        pbar.write(f"    查询状态失败 ({consecutive_errors}/{self.max_consecutive_errors}), 继续重试...")
                        continue
                    
                    # 重置错误计数
                    consecutive_errors = 0
                    
                    status = response.json()
                    state = status.get('state')
                    progress = status.get('progress', 0)
                    message = status.get('message', '')
                    
                    # 更新进度条
                    if progress > last_progress:
                        pbar.update(progress - last_progress)
                        last_progress = progress
                    
                    # 显示当前阶段信息
                    if message:
                        pbar.set_postfix_str(message)
                    
                    # 检查任务状态
                    if state == 'completed':
                        pbar.update(100 - last_progress)  # 确保到100%
                        result = status.get('result')
                        if result is None:
                            raise Exception("任务完成但未返回结果")
                        return result
                    
                    elif state == 'failed':
                        error = status.get('error', '未知错误')
                        raise Exception(f"服务端处理失败: {error}")
                    
                    elif state in ['pending', 'processing']:
                        # 如果是pending状态，显示队列信息
                        if state == 'pending':
                            queue_pos = status.get('queue_position', 0)
                            wait_time = status.get('estimated_wait_time', 0)
                            wait_minutes = int(wait_time) // 60
                            pbar.set_postfix_str(f"队列位置: {queue_pos}, 等待: {wait_minutes}分钟")
                        continue
                    
                    else:
                        pbar.write(f"    未知状态: {state}, 继续等待...")
                        continue
                
                except requests.exceptions.Timeout:
                    consecutive_errors += 1
                    if consecutive_errors >= self.max_consecutive_errors:
                        raise Exception(f"连续{self.max_consecutive_errors}次查询超时")
                    pbar.write(f"    查询超时 ({consecutive_errors}/{self.max_consecutive_errors}), 继续重试...")
                    continue
                
                except requests.exceptions.ConnectionError:
                    consecutive_errors += 1
                    if consecutive_errors >= self.max_consecutive_errors:
                        raise Exception("无法连接到服务")
                    pbar.write(f"    连接失败 ({consecutive_errors}/{self.max_consecutive_errors}), 继续重试...")
                    continue
            
            # 超过最大等待时间
            raise Exception(f"任务超时 (超过 {self.max_wait_time / 3600:.1f} 小时)")


    def check_queue_status(self):
        """
        检查队列状态
        """
        try:
            response = requests.get(
                f'{self.video_api_base}/video_handler/queue/status',
                timeout=10
            )
            
            if response.status_code == 200:
                status = response.json()
                return status
            else:
                print(f"    队列状态查询失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"    队列状态查询异常: {e}")
            return None


    def get_queue_summary(self):
        """
        获取队列摘要信息
        """
        status = self.check_queue_status()
        if status:
            counts = status.get('status_counts', {})
            queue_info = status.get('queue_info', {})
            current_task = status.get('current_task', {})
            
            summary = {
                'pending': counts.get('pending', 0),
                'processing': counts.get('processing', 0),
                'completed': counts.get('completed', 0),
                'failed': counts.get('failed', 0),
                'queue_size': queue_info.get('current_size', 0),
                'max_queue_size': queue_info.get('max_size', 0),
                'current_task': current_task
            }
            return summary
        return None


    def analyze_video(self, video_path, video_info):
        """
        调用接口处理视频 (队列模式)
        """
        video_id = video_info['content_id']
        print(f"    开始处理视频 {video_id}...")
        
        try:
            # 步骤1: 提交视频任务到队列
            submit_result = self._submit_video_task(video_path, video_info)
            task_id = submit_result['task_id']
            queue_position = submit_result['queue_position']
            wait_minutes = submit_result['estimated_wait_time'] // 60
            
            print(f"    任务已提交 (ID: {task_id}), 视频ID: {video_id}")
            print(f"    队列位置: {queue_position}, 预计等待: {wait_minutes}分钟")
            
            # 步骤2: 等待任务完成
            result = self._wait_for_task_completion(task_id)
            print(f"    处理完成, 视频ID: {video_id}")
            return result
                
        except FileNotFoundError:
            error_msg = f"视频文件不存在: {video_path}"
            print(f"    {error_msg}")
            raise Exception(error_msg)
        
        except Exception as e:
            error_msg = f"处理失败, 视频ID: {video_id}: {str(e)}"
            print(f"    {error_msg}")
            raise Exception(error_msg)
    

    def get_videos(self, keyword=None, start_time=None, end_time=None, only_unprocessed=False):
        """
        根据关键词和时间范围获取视频列表
        """
        # 连接数据库
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 构建查询
        sql = """
            SELECT 
                content_id,
                collect_keyword,
                video_url,
                create_time,
                del_flag,
                analysis_summary
            FROM collect_content
            WHERE content_type = '0'
        """
        
        params = []
        
        # 只获取未处理的视频
        if only_unprocessed:
            sql += " AND del_flag != '1'"
            sql += " AND (analysis_summary IS NULL OR analysis_summary = '' OR analysis_summary = '[]')"
        
        # 提供了关键词
        if keyword:
            sql += " AND collect_keyword LIKE %s"
            params.append(f'%{keyword}%')

        # 处理开始时间
        if start_time:
            if isinstance(start_time, str):
                # 解析多种格式
                if len(start_time) == 10:  # "2025-10-20"
                    start_time = datetime.strptime(start_time, "%Y-%m-%d")
                elif len(start_time) == 19:  # "2025-10-20 09:35:36"
                    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                else:
                    raise ValueError(f"不支持的时间格式: {start_time}")
            
            sql += " AND create_time >= %s"
            params.append(start_time)
        
        # 处理结束时间
        if end_time:
            if isinstance(end_time, str):
                if len(end_time) == 10:  # "2025-10-20"
                    # 将日期转换为当天的 23:59:59
                    end_time = datetime.strptime(end_time, "%Y-%m-%d")
                    end_time = datetime.combine(end_time.date(), time_module(23, 59, 59))
                elif len(end_time) == 19:  # "2025-10-20 09:35:36"
                    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                else:
                    raise ValueError(f"不支持的时间格式: {end_time}")
            
            sql += " AND create_time <= %s"
            params.append(end_time)
        
        sql += " ORDER BY create_time DESC"
        
        # 执行查询
        cursor.execute(sql, params)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return results
    

    def analyze_video_query_text(self, query_text, video_id):
        """
        分析视频的query文本内容（调用推文分析服务）
        """
        if not query_text or not query_text.strip():
            return None
        
        try:
            print(f"    开始分析视频query文本...")
            analysis_result = analyze_single_tweet_text(
                self.tweet_analyze_base_url,
                query_text,
                video_id
            )
            
            if analysis_result and analysis_result.get('txt_analysis_json'):
                print(f"    推文分析完成")
                return analysis_result.get('txt_analysis_json')
            else:
                print(f"    推文分析失败或无结果")
                return None
                
        except Exception as e:
            print(f"    推文分析异常: {e}")
            return None


    def save_results_to_db(self, video_id, result, txt_analysis_json=None):
        """
        将处理结果保存到数据库
        """
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # 获取del_flag
            del_flag = result.get('del_flag', '0')

            # 初始化入库变量
            query_content = None
            analysis_tag = None
            video_analysis_json = None
            final_txt_analysis_json = None
            analysis_summary = None
            analysis_search = None

            # del_flag为1，所有值保持为None
            if del_flag == '1':
                print(f"    检测到del_flag='1'，所有分析结果为空")
            else:
                query_content = result.get('query')

                # 处理 txt_analysis_json
                if txt_analysis_json is not None:
                    if isinstance(txt_analysis_json, list):
                        final_txt_analysis_json = json.dumps(txt_analysis_json, ensure_ascii=False)
                    else:
                        final_txt_analysis_json = txt_analysis_json
                
                # 提取标签
                analysis_data = []
                if isinstance(txt_analysis_json, str):
                    try:
                        analysis_data = json.loads(txt_analysis_json)
                    except json.JSONDecodeError:
                        analysis_data = []
                elif isinstance(txt_analysis_json, list):
                    analysis_data = txt_analysis_json

                # 基础标签
                china_related = ""
                sentiment = ""
                classification = ""

                if analysis_data:
                    for item in analysis_data:
                        key = item.get('key')
                        value = item.get('value', '')
                        
                        if key == '是否涉华':
                            if value: 
                                china_related = f'是否涉华-{value}'
                        elif key == '情感倾向':
                            if value:
                                sentiment = f'情感倾向-{value}'
                        elif key == '推文涉华情况分类':
                            if value:
                                classification = f'涉华情况分类-{value}'

                # 动态组合前缀
                prefix_list = []
                if china_related:
                    prefix_list.append(china_related)
                if sentiment:
                    prefix_list.append(sentiment)
                if classification:
                    prefix_list.append(classification)
                
                # 提取 elements 标签
                elements_list = []
                if result.get('elements'):
                    for item in result['elements']:
                        if item.get('elements'):
                            elements_list.extend(item['elements'])
                
                unique_elements = list(set(elements_list))
                
                # 组合最终标签字符串
                final_tag_parts = prefix_list + unique_elements
                analysis_tag = ','.join(final_tag_parts) if final_tag_parts else None
                            
                # 构造其他 JSON 字段
                if result.get('elements'):
                    video_analysis_json = json.dumps(result['elements'], ensure_ascii=False)
                
                if result.get('summary'):
                    analysis_summary = result.get('summary')

                if result.get('search'):
                    analysis_search = json.dumps(result['search'], ensure_ascii=False)

            # 更新collect_content表
            update_sql = """
                UPDATE collect_content 
                SET analysis_content = %s,
                    analysis_tag = %s,
                    video_analysis_json = %s,
                    txt_analysis_json = %s,
                    analysis_summary = %s,
                    analysis_search = %s,
                    del_flag = %s
                WHERE content_id = %s
            """
            cursor.execute(update_sql, (
                query_content,
                analysis_tag,
                video_analysis_json,
                final_txt_analysis_json,
                analysis_summary,
                analysis_search,
                str(del_flag),
                video_id
            ))
            
            conn.commit()
            print(f"    数据库更新成功")
            return True
            
        except Exception as e:
            conn.rollback()
            print(f"    数据库更新失败: {e}")
            return False
            
        finally:
            cursor.close()
            conn.close()


    def convert_path_to_url(self, local_path):
        """
        将本地路径转换为下载URL
        """
        local_path = local_path.replace('\\', '/')
        if local_path.startswith(self.base_local_path):
            relative_path = local_path.replace(self.base_local_path, '').lstrip('/')
            download_url = f"{self.base_url}/{relative_path}"
            return download_url
        else:
            raise ValueError(f"路径格式不正确: {local_path}")

    
    def get_video_duration(self, url):
        """
        使用 ffprobe 获取视频时长(秒)，失败返回一个较大值(保证排序靠后)
        """
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nk=1:nw=1",
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            duration = float(result.stdout.strip())
            return duration
        except:
            return float('inf')
    

    def download_video(self, url, save_path):
        """
        使用wget下载视频
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # wget下载
            cmd = [
                'wget',
                '-q',
                '-t', '1',
                '--timeout=60',  # 超时60秒
                '-O', save_path,
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                if file_size > 0:
                    print(f"    下载成功 ({file_size / 1024 / 1024:.2f} MB)")
                    return True
                else:
                    print(f"    下载失败: 文件为空")
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    return False
            else:
                print(f"    下载失败: wget返回码 {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"    下载超时 (>5分钟)")
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
            
        except Exception as e:
            print(f"    下载异常: {e}")
            return False
    

    def process_videos(self, keyword=None, start_time=None, end_time=None, save_json=False, save_to_db=True, only_unprocessed=False):
        """
        逐个处理视频：下载 → 处理 → 存储到数据库 → 删除
        """
        print("\n" + "="*80)
        print(f"开始处理视频")
        print(f"关键词: {keyword if keyword else '全部'}")
        print(f"时间范围: {start_time or '不限'} ~ {end_time or '不限'}")
        print(f"处理模式: {'仅未处理' if only_unprocessed else '全部'}")
        print(f"保存到数据库: {'是' if save_to_db else '否'}")
        # 检查队列状态
        queue_summary = self.get_queue_summary()
        if queue_summary:
            print(f"当前队列: {queue_summary['pending']}等待, {queue_summary['processing']}处理中")
            if queue_summary['pending'] > 10:
                print(f"队列较长，预计等待时间较长")
        print("="*80 + "\n")
        
        # 1. 获取视频列表
        videos = self.get_videos(keyword, start_time, end_time, only_unprocessed)
        
        if not videos:
            print("未找到符合条件的视频")
            return
        else:
            print("正在获取视频时长以进行排序(从短到长)...")
            for v in videos:
                try:
                    url = self.convert_path_to_url(v['video_url'])
                    v['duration'] = self.get_video_duration(url)
                except:
                    v['duration'] = float('inf')
            videos.sort(key=lambda x: x['duration'])
        
        total = len(videos)
        print(f"共找到 {total} 个视频\n")
        
        # 统计
        success_count = 0
        download_fail_count = 0
        analysis_fail_count = 0
        db_save_fail_count = 0

        # 最终输出结果（不含错误信息）
        all_results = {}
        
        # 内部追踪失败信息
        failed_videos = {}
        
        # 2. 逐个处理
        for idx, video in enumerate(videos, 1):
            video_id = video['content_id']
            video_url_path = video['video_url']
            local_file = os.path.join(self.download_dir, f"{video_id}_video.mp4")
            
            print(f"[{idx}/{total}] {video_id}")

            # 初始化视频结果（原始格式，不含错误字段）
            all_results[video_id] = {
                'keyword': video['collect_keyword'],
                'video_url': video['video_url'],
                'create_time': str(video['create_time']),
                'query': None,
                'elements': None,
                'summary': None,
                'search': None,
                'txt_analysis': None,
                'del_flag': '0',
                'segments': None   
            }
            
            try:
                # Step 1: 下载
                download_url = self.convert_path_to_url(video_url_path)
                print(f"  URL: {download_url}")
                
                if not self.download_video(download_url, local_file):
                    download_fail_count += 1
                    # 记录到失败列表
                    failed_videos[video_id] = {
                        **all_results[video_id],
                        'error_type': 'download_failed',
                        'error_msg': f'下载失败'
                    }
                    print()
                    continue
                
                # Step 2: 处理
                try:
                    result = self.analyze_video(local_file, video)
                    # 获取 del_flag
                    del_flag = result.get('del_flag', '0')
                    all_results[video_id]['query'] = result.get('query', None)
                    all_results[video_id]['elements'] = result.get('elements', None)
                    all_results[video_id]['summary'] = result.get('summary', None)
                    all_results[video_id]['search'] = result.get('search', None)
                    all_results[video_id]['segments'] = result.get('segments', None)
                    all_results[video_id]['del_flag'] = del_flag

                    # Step 2.5: 分析视频query文本（调用推文分析服务）
                    txt_analysis_json = None
                    if del_flag == '1':
                        print(f"    del_flag='1'，跳过推文文本分析")
                    else:
                        query_text = result.get('query', None)
                        if query_text:
                            txt_analysis_json = self.analyze_video_query_text(query_text, video_id)
                    all_results[video_id]['txt_analysis'] = txt_analysis_json  

                    # Step 3: 保存到数据库
                    if save_to_db:
                        if self.save_results_to_db(video_id, result, txt_analysis_json):
                            success_count += 1
                        else:
                            db_save_fail_count += 1
                            failed_videos[video_id] = {
                                **all_results[video_id],
                                'error_type': 'db_save_failed',
                                'error_msg': '数据库保存失败'
                            }
                    else:
                        success_count += 1
                    
                except Exception as e:
                    print(f"    处理最终失败: {e}")
                    analysis_fail_count += 1
                    # 记录到失败列表
                    failed_videos[video_id] = {
                        **all_results[video_id],
                        'error_type': 'analysis_failed',
                        'error_msg': f'处理失败: {str(e)}'
                    }
                
            except Exception as e:
                print(f"    处理异常: {e}")
                download_fail_count += 1
                # 记录到失败列表
                failed_videos[video_id] = {
                    **all_results[video_id],
                    'error_type': 'exception',
                    'error_msg': f'异常: {str(e)}'
                }
            
            finally:
                # Step 3: 删除临时文件
                if os.path.exists(local_file):
                    try:
                        os.remove(local_file)
                        print(f"    已删除临时文件")
                    except Exception as e:
                        print(f"    删除失败: {e}")
            
            print()
        
        # 3. 输出统计
        print("="*80)
        print("处理完成")
        print(f"总数: {total}")
        print(f"  成功: {success_count}")
        print(f"  下载失败: {download_fail_count}")
        print(f"  处理失败: {analysis_fail_count}")
        if save_to_db:
            print(f"  数据库保存失败: {db_save_fail_count}")
        print(f"成功率: {success_count/total*100:.1f}%")
        print("="*80 + "\n")

        # 4. 保存结果
        if save_json and all_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            keyword_str = keyword if keyword else 'all'

            # 保存完整结果（不含错误信息）
            json_file = os.path.join(
                self.download_dir, 
                f"video_report_{keyword_str}_{timestamp}.json"
            )
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            print(f"结果已保存到: {json_file}")
            
            # 如果有失败的，单独保存失败列表（含错误信息）
            if failed_videos:
                failed_file = os.path.join(
                    self.download_dir,
                    f"video_failed_{keyword_str}_{timestamp}.json"
                )
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_videos, f, ensure_ascii=False, indent=2)
                
                print(f"失败列表已保存到: {failed_file} ({len(failed_videos)}个)")
                
        return all_results


    def check_gpu_memory_usage(self):
        """检查GPU 0的显存使用率"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # 按行分割,取第一行(GPU 0)
                lines = result.stdout.strip().split('\n')
                
                if lines:
                    line = lines[0].strip()  # GPU 0
                    # 按逗号分割,并去除空格
                    parts = [p.strip() for p in line.split(',')]
                    
                    if len(parts) >= 2:
                        used = float(parts[0])
                        total = float(parts[1])
                        usage_percent = (used / total) * 100
                        print(f"    [GPU0] {used:.0f}MB / {total:.0f}MB ({usage_percent:.1f}%)")
                        return usage_percent
                
                print(f"    [警告] nvidia-smi输出格式异常")
                return 0
            else:
                print(f"    [警告] nvidia-smi返回错误码: {result.returncode}")
                return 0
                
        except Exception as e:
            print(f"    [警告] 无法获取显存信息: {e}")
            return 0


    def process_videos_smart(self, memory_threshold=70.0, **kwargs):
        """
        智能处理视频
        Args:
            memory_threshold: 显存占用阈值(百分比)，超过此值将重启服务
            **kwargs: 视频筛选条件 (keyword, start_time, end_time, only_unprocessed)
        """        
        # 初始化服务管理器
        service_manager = TmuxServiceManager(session_name="video_services")
        
        # 检查tmux会话
        if not service_manager.check_tmux_session():
            print(f"错误: tmux会话 '{service_manager.session_name}' 不存在!")
            return
        
        # 确保服务运行
        if not service_manager.is_main_service_running():
            print("main服务未运行，正在启动...")
            if not service_manager.start_main_service():
                print("无法启动main服务，退出")
                return
            if not service_manager.wait_for_service_ready():
                print("服务启动失败，退出")
                return
        
        # 获取视频列表
        videos = self.get_videos(
            keyword=kwargs.get('keyword'),
            start_time=kwargs.get('start_time'),
            end_time=kwargs.get('end_time'),
            only_unprocessed=kwargs.get('only_unprocessed', False)
        )
        
        if not videos:
            print("未找到符合条件的视频")
            return
        
        total = len(videos)
        
        print(f"\n{'='*80}")
        print(f"智能处理模式: 共 {total} 个视频")
        print(f"显存阈值: {memory_threshold}% (超过将自动重启服务)")
        print(f"{'='*80}\n")
        
        # 进度文件
        progress_file = os.path.join(self.download_dir, '.smart_progress.json')
        
        # 读取进度
        start_idx = 0
        restart_count = 0
        success_count = 0
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    start_idx = progress.get('next_video', 0)
                    restart_count = progress.get('restart_count', 0)
                    success_count = progress.get('success_count', 0)
                    print(f"从第 {start_idx + 1} 个视频继续处理")
                    print(f"已成功: {success_count}, 已重启: {restart_count} 次\n")
            except:
                pass
        
        # 主处理循环
        for idx in range(start_idx, total):
            video = videos[idx]
            
            print(f"\n{'='*80}")
            print(f"[{idx + 1}/{total}] 处理视频")
            print(f"{'='*80}\n")
            
            # 处理单个视频
            if self._process_single_video(video, idx + 1):
                success_count += 1
            
            # 保存进度
            with open(progress_file, 'w') as f:
                json.dump({
                    'next_video': idx + 1,
                    'restart_count': restart_count,
                    'success_count': success_count
                }, f)
            
            # 如果不是最后一个视频,检查是否需要重启
            if idx < total - 1:
                print(f"\n  [检查] 显存使用情况:")
                memory_usage = self.check_gpu_memory_usage()
                
                if memory_usage >= memory_threshold:
                    print(f"\n  [警告] 显存占用 {memory_usage:.1f}% >= 阈值 {memory_threshold}%")
                    print(f"  [动作] 准备重启服务释放显存...\n")
                    
                    restart_count += 1
                    
                    if service_manager.restart_main_service():
                        print(f"  [成功] 服务重启完成 (第 {restart_count} 次重启)")
                        time.sleep(5)
                    else:
                        print(f"  [失败] 服务重启失败，终止处理")
                        break
                else:
                    print(f"  [正常] 显存占用 {memory_usage:.1f}% < 阈值 {memory_threshold}%，继续处理")
        
        # 删除进度文件
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        # 最终统计
        print(f"\n{'='*80}")
        print(f"处理完成!")
        print(f"{'='*80}")
        print(f"总视频数: {total}")
        print(f"成功处理: {success_count}")
        print(f"失败数量: {total - success_count}")
        print(f"服务重启: {restart_count} 次")
        print(f"{'='*80}\n")


    def _process_single_video(self, video, video_num):
        """
        处理单个视频
        """
        video_id = video['content_id']
        video_url_path = video['video_url']
        local_file = os.path.join(self.download_dir, f"{video_id}_video.mp4")
        
        print(f"视频ID: {video_id}")

        # 文件路径
        master_json_file = os.path.join(self.download_dir, "video_report.json")
        failed_master_file = os.path.join(self.download_dir, "video_failed.json")

        # 数据结构
        current_result = {
            'keyword': video['collect_keyword'],
            'video_url': video['video_url'],
            'create_time': str(video['create_time']),
            'query': None,
            'elements': None,
            'summary': None,
            'search': None,
            'txt_analysis': None,
            'del_flag': '0',
            'segments': None   
        }
        
        success = False
        
        try:
            # 下载
            try:
                download_url = self.convert_path_to_url(video_url_path)
            except ValueError as e:
                # 路径格式错误，标记为需要删除
                current_result['del_flag'] = '1'
                raise Exception(f"路径格式错误: {e}")
            
            print(f"  [下载] {download_url}")

            if not self.download_video(download_url, local_file):
                raise Exception("下载失败")
            
            # 视频分析
            print(f"  [分析] 开始视频分析...")
            analysis_data = self.analyze_video(local_file, video)

            if current_result['del_flag'] == '0':
                del_flag = analysis_data.get('del_flag', '0')
            else:
                del_flag = current_result['del_flag']
                analysis_data['del_flag'] = del_flag

            current_result.update({
                'query': analysis_data.get('query'),
                'elements': analysis_data.get('elements'),
                'summary': analysis_data.get('summary'),
                'search': analysis_data.get('search'),
                'segments': analysis_data.get('segments'),
                'del_flag': del_flag
            })                    
                
            # 文本分析
            txt_analysis_json = None
            if del_flag != '1' and current_result['query']:
                print(f"  [分析] 开始文本分析...")
                txt_analysis_json = self.analyze_video_query_text(
                    current_result['query'], 
                    video_id
                )
            current_result['txt_analysis'] = txt_analysis_json                   
            
            # 保存到数据库
            if not self.save_results_to_db(video_id, analysis_data, txt_analysis_json):
                raise Exception("数据库保存失败")

            # 保存到JSON
            print(f"  [保存] 写入结果文件...")
            self.append_dict_to_json(master_json_file, {video_id: current_result})
            print(f"  [成功] 视频处理完成")
            success = True

        except Exception as e:
            print(f"  [错误] {e}")
            error_info = {
                **current_result,
                'error_type': 'failed',
                'error_msg': str(e)
            }
            self.append_dict_to_json(failed_master_file, {video_id: error_info})
        
        finally:
            # 清理文件
            if os.path.exists(local_file):
                try:
                    os.remove(local_file)
                    print(f"  [清理] 删除本地文件")
                except:
                    pass

            # 清理内存
            if 'analysis_data' in locals(): 
                del analysis_data
            if 'txt_analysis_json' in locals(): 
                del txt_analysis_json
            if 'current_result' in locals(): 
                del current_result
            
            gc.collect()
            print(f"  [清理] 内存清理完成")
        
        return success


    def append_dict_to_json(self, filename, new_data_dict):
        """
        将新的字典数据追加/更新到 JSON 文件
        - 同一 video_id，新数据覆盖旧数据
        - 保证每个 video_id 只出现一次
        """
        if not new_data_dict:
            return

        # 如果文件不存在，直接创建并写入
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(new_data_dict, f, ensure_ascii=False, indent=4)
            print(f"    [保存] 新建文件并写入 {len(new_data_dict)} 条数据")
            return

        # 读取现有数据
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"    [警告] JSON格式错误，将重新创建: {e}")
            existing_data = {}

        # 统计更新和新增
        updated_count = sum(1 for k in new_data_dict if k in existing_data)
        new_count = sum(1 for k in new_data_dict if k not in existing_data)
        
        # 更新数据（新数据覆盖旧数据）
        existing_data.update(new_data_dict)
        
        # 写回文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        
        # 输出统计
        total = len(existing_data)
        if updated_count > 0 and new_count > 0:
            print(f"    [保存] 覆盖 {updated_count} 条，新增 {new_count} 条，总共 {total} 条记录")
        elif updated_count > 0:
            print(f"    [保存] 覆盖 {updated_count} 条，总共 {total} 条记录")
        else:
            print(f"    [保存] 新增 {new_count} 条，总共 {total} 条记录")


if __name__ == "__main__":
    analyzer = VideoAnalyzer()

    # analyzer.process_videos(
    #     keyword="特朗普",
    #     start_time=datetime(2025, 10, 28, 1, 42, 0),
    #     end_time=datetime(2025, 10, 28, 1, 43, 30)
    # )

    # analyzer.process_videos()

    try:
        analyzer.process_videos_smart(
            memory_threshold=65.0,
            keyword=None,
            start_time=None,
            end_time=None,
            only_unprocessed=True
        )
        
        print("\n脚本执行完成")
        sys.exit(0)
        
    except KeyboardInterrupt:
        # 用户按 Ctrl+C
        print("\n用户中断执行")
        sys.exit(130)  # 标准的中断退出码
        
    except Exception as e:
        print(f"\n致命错误: {e}")
        traceback.print_exc()
        sys.exit(1)