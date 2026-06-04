import asyncio
import subprocess
import time
import traceback
import pymysql
import os
import signal
import sys
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DB


# 数据库配置
DB_CONFIG = {
    "host": MYSQL_HOST,
    "user": MYSQL_USER,
    "port": MYSQL_PORT,
    "password": MYSQL_PWD,
    "database": MYSQL_DB,
    'charset': 'utf8mb4'
}

# 脚本配置
SCRIPTS = {
    'collect_content': {
        'script_path': '/data/srch_vt/src/run_video.py',
        'script_name': 'run_video.py',
        'priority': 2,
        'check_sql': """
            SELECT COUNT(*) as cnt
            FROM collect_content
            WHERE content_type = '0'
              AND del_flag = '0'
              AND (analysis_summary IS NULL OR analysis_summary = '' OR analysis_summary = '[]')
        """
    },
    'anchor_zb_video': {
        'script_path': '/data/srch_vt/src/run_zb.py',
        'script_name': 'run_zb.py',
        'priority': 1,
        'check_sql': """
            SELECT COUNT(*) as cnt
            FROM anchor_zb_video
            WHERE del_flag = '0'
              AND (analysis_summary IS NULL OR analysis_summary = '' OR analysis_summary = '[]')
        """
    }
}

# 扫描间隔(秒)
SCAN_INTERVAL = 120  # 2分钟扫描一次

# 工作目录
WORK_DIR = Path(__file__).parent / '.scanner_work'
WORK_DIR.mkdir(exist_ok=True)


# 进程检查器
class ProcessChecker:
    """检查目标脚本是否正在运行"""
    @staticmethod
    def is_script_running(script_names: List[str]) -> tuple[bool, Optional[str]]:
        """
        检查指定的脚本是否正在运行
        Args:
            script_names: 脚本名列表，如 ['run_video.py', 'run_zb.py']
        Returns:
            (是否运行, 运行中的脚本名)
        """
        try:
            current_pid = os.getpid()
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # 跳过自己
                    if proc.info['pid'] == current_pid:
                        continue
                    
                    cmdline = proc.info.get('cmdline')
                    if not cmdline:
                        continue
                    
                    # 将命令行参数合并成字符串
                    cmdline_str = ' '.join(cmdline)
                    
                    # 检查是否包含目标脚本名
                    for script_name in script_names:
                        if script_name in cmdline_str:
                            # 确认是 python 进程
                            if 'python' in cmdline_str.lower():
                                return True, script_name
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            return False, None
            
        except Exception as e:
            print(f"[警告] 进程检查异常: {e}")
            return False, None
    

    @staticmethod
    def get_process_info(script_name: str) -> Optional[Dict]:
        """获取指定脚本的进程信息"""
        try:
            current_pid = os.getpid()
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
                try:
                    if proc.info['pid'] == current_pid:
                        continue
                    
                    cmdline = proc.info.get('cmdline')
                    if not cmdline:
                        continue
                    
                    cmdline_str = ' '.join(cmdline)
                    
                    if script_name in cmdline_str and 'python' in cmdline_str.lower():
                        create_time = datetime.fromtimestamp(proc.info['create_time'])
                        running_time = datetime.now() - create_time
                        
                        memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                        
                        return {
                            'pid': proc.info['pid'],
                            'cmdline': cmdline_str,
                            'start_time': create_time,
                            'running_time': running_time,
                            'memory_mb': memory_mb
                        }
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            return None
            
        except Exception as e:
            print(f"[警告] 获取进程信息异常: {e}")
            return None


# 扫描器
class SimpleScanner:
    """进程感知扫描器 - 检测进程后再扫描"""
    def __init__(self, db_config: Dict, scripts: Dict):
        self.db_config = db_config
        self.scripts = scripts
        self.stop_event = asyncio.Event()
        self.process_checker = ProcessChecker()
        # 提取所有脚本名
        self.script_names = [config['script_name'] for config in scripts.values()]
    

    def get_db_connection(self):
        return pymysql.connect(**self.db_config)
    

    def check_table(self, table_key: str) -> int:
        """检查表中是否有待处理记录"""
        script_config = self.scripts[table_key]
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute(script_config['check_sql'])
            result = cursor.fetchone()
            count = result[0] if result else 0
            cursor.close()
            conn.close()
            return count
        except Exception as e:
            print(f"[错误] 检查 {table_key} 失败: {e}")
            return 0
    

    async def run_script(self, table_key: str, count: int) -> bool:
        """
        运行处理脚本
        """
        script_config = self.scripts[table_key]
        script_path = script_config['script_path']
        
        # 检查脚本是否存在
        if not os.path.exists(script_path):
            print(f"[错误] 脚本不存在: {script_path}")
            return False
        
        print(f"\n{'='*70}")
        print(f"[执行] {table_key}")
        print(f"[脚本] {os.path.basename(script_path)}")
        print(f"[待处理] {count} 条记录")
        print(f"{'='*70}\n")
        
        process = None
        
        try:
            start_time = time.time()
            
            # 启动子进程
            process = await asyncio.create_subprocess_exec(
                'python', script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(os.path.abspath(script_path)) or '.'
            )
            
            # 实时输出脚本日志
            async def stream_output(stream, prefix):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    print(f"{prefix}{line.decode('utf-8', errors='ignore').rstrip()}")
            
            # 并发读取stdout和stderr
            await asyncio.gather(
                stream_output(process.stdout, "  "),
                stream_output(process.stderr, "  [ERR] ")
            )
            
            # 等待进程结束
            await process.wait()
            
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            
            print(f"\n{'='*70}")
            
            # 根据返回码判断结果
            if process.returncode == 0:
                print(f"脚本执行成功")
                print(f"   返回码: {process.returncode}")
                print(f"   耗时: {mins}分{secs}秒")
                result = True
            
            elif process.returncode == 130:  # SIGINT (Ctrl+C)
                print(f"脚本被用户中断")
                print(f"   返回码: {process.returncode}")
                print(f"   耗时: {mins}分{secs}秒")
                result = False
            
            elif process.returncode < 0:  # 被信号杀死
                signal_num = -process.returncode
                print(f"脚本被信号终止")
                print(f"   返回码: {process.returncode} (信号 {signal_num})")
                print(f"   耗时: {mins}分{secs}秒")
                result = False
            
            else:  # 其他非0返回码
                print(f"脚本执行失败")
                print(f"   返回码: {process.returncode}")
                print(f"   耗时: {mins}分{secs}秒")
                result = False
            
            print(f"{'='*70}\n")
            
            return result
            
        except asyncio.CancelledError:
            # Scanner 被取消时的处理
            print(f"\nScanner 被取消，正在终止子进程...")
            
            if process and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                    print("   子进程已终止")
                except asyncio.TimeoutError:
                    print("   强制杀死子进程...")
                    process.kill()
                    await process.wait()
                except Exception as e:
                    print(f"   终止子进程时出错: {e}")
            
            raise
        
        except Exception as e:
            print(f"\n执行异常: {e}")
            traceback.print_exc()
            
            # 尝试清理子进程
            if process and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass
            
            return False
    

    def check_running_scripts(self) -> tuple[bool, Optional[str], Optional[Dict]]:
        """
        检查是否有目标脚本正在运行
        Returns:
            (是否有脚本运行, 脚本名, 进程信息)
        """
        is_running, script_name = self.process_checker.is_script_running(self.script_names)
        
        if is_running and script_name:
            proc_info = self.process_checker.get_process_info(script_name)
            return True, script_name, proc_info
        
        return False, None, None
    
    
    async def scan_once(self):
        """执行一次扫描"""
        # 在扫描开始时检查停止标志
        if self.stop_event.is_set():
            return

        print(f"\n{'='*70}")
        print(f"扫描周期开始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # 第一步：检查进程
        print(f"[步骤1] 检查目标脚本进程状态...")
        is_running, script_name, proc_info = self.check_running_scripts()
        
        if is_running:
            print(f"发现 {script_name} 正在运行中")
            
            if proc_info:
                running_mins = int(proc_info['running_time'].total_seconds() / 60)
                print(f"   PID: {proc_info['pid']}")
                print(f"   运行时长: {running_mins} 分钟")
                print(f"   内存占用: {proc_info['memory_mb']:.1f} MB")
                print(f"   启动时间: {proc_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            print(f"\n[跳过] 等待脚本完成后再扫描")
            print(f"{'='*70}\n")
            return
        
        print(f"无目标脚本运行\n")
        
        # 第二步：扫描所有数据库表
        print(f"[步骤2] 扫描所有数据库表...")
        
        # 收集所有待处理任务
        pending_tasks = []
        
        for table_key, config in self.scripts.items():
            # 在循环中检查停止标志
            if self.stop_event.is_set():
                print(f"\n[中断] 检测到停止信号")
                return

            try:
                print(f"  [检查] {table_key}...", end=' ')
                count = self.check_table(table_key)
                
                if count > 0:
                    print(f"发现 {count} 条待处理记录")
                    pending_tasks.append({
                        'table_key': table_key,
                        'count': count,
                        'priority': config.get('priority', 999)
                    })
                else:
                    print("无待处理记录")
                    
            except Exception as e:
                print(f"错误: {e}")
        
        # 第三步：如果有待处理任务，按优先级依次执行所有任务
        if pending_tasks:
            # 按优先级排序（数字越小优先级越高）
            pending_tasks.sort(key=lambda x: x['priority'])
            
            print(f"\n[汇总] 发现 {len(pending_tasks)} 个表有待处理记录:")
            for task in pending_tasks:
                print(f"  - {task['table_key']}: {task['count']} 条 (优先级: {task['priority']})")
            
            # ===== 关键修改：依次处理所有表 =====
            print(f"\n[步骤3] 依次处理所有表\n")
            
            for idx, task in enumerate(pending_tasks, 1):
                # 检查停止标志
                if self.stop_event.is_set():
                    print(f"\n[中断] 检测到停止信号，已处理 {idx-1}/{len(pending_tasks)} 个表")
                    return
                
                table_key = task['table_key']
                count = task['count']
                priority = task['priority']
                
                print(f"\n{'='*70}")
                print(f"[处理 {idx}/{len(pending_tasks)}] {table_key}")
                print(f"[优先级] {priority}")
                print(f"[待处理] {count} 条记录")
                print(f"{'='*70}\n")
                
                try:
                    # 执行脚本
                    success = await self.run_script(table_key, count)
                    
                    if success:
                        print(f"\n[{idx}/{len(pending_tasks)}] {table_key} 处理完成")
                    else:
                        print(f"\n[{idx}/{len(pending_tasks)}] {table_key} 处理失败")
                    
                except asyncio.CancelledError:
                    print(f"\n[取消] 任务被取消")
                    raise
                except Exception as e:
                    print(f"\n[{idx}/{len(pending_tasks)}] {table_key} 处理异常: {e}")
                    # 继续处理下一个表
                    continue
                
                # 处理完一个表后，短暂等待（避免数据库压力过大）
                if idx < len(pending_tasks):
                    wait_seconds = 5
                    print(f"\n[等待] {wait_seconds}秒后处理下一个表...")
                    
                    try:
                        # 使用可中断的等待
                        await asyncio.wait_for(
                            self.stop_event.wait(),
                            timeout=wait_seconds
                        )
                        # 如果收到停止信号，退出
                        print(f"\n[中断] 检测到停止信号")
                        return
                    except asyncio.TimeoutError:
                        # 超时正常，继续下一个表
                        pass
            
            print(f"\n{'='*70}")
            print(f"本次扫描完成 - 共处理 {len(pending_tasks)} 个表")
            print(f"{'='*70}\n")
        else:
            print(f"\n[结果] 本次扫描未发现待处理记录")
            print(f"{'='*70}\n")
    

    async def run(self):
        """运行扫描循环"""        
        print(f"\n{'*'*70}")
        print(f"进程感知扫描器已启动")
        print(f"{'*'*70}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"监控表: {', '.join(self.scripts.keys())}")
        print(f"监控脚本: {', '.join(self.script_names)}")
        print(f"扫描间隔: {SCAN_INTERVAL}秒 ({SCAN_INTERVAL//60}分钟)")
        print(f"{'*'*70}\n")
        
        try:
            # 首次扫描
            await self.scan_once()
        
            # 循环扫描
            while not self.stop_event.is_set():
                # 计算下次扫描时间
                next_scan = datetime.now() + timedelta(seconds=SCAN_INTERVAL)
                print(f"等待 {SCAN_INTERVAL//60} 分钟后进行下一次扫描...")
                print(f"   下次扫描时间: {next_scan.strftime('%Y-%m-%d %H:%M:%S')}")
                
                try:
                    await asyncio.wait_for(
                        self.stop_event.wait(),
                        timeout=SCAN_INTERVAL
                    )
                    # 如果 wait 返回了（没超时），说明收到停止信号
                    print("\n[信号] 收到停止信号，退出扫描循环")
                    break

                except asyncio.TimeoutError:
                    # 继续下一次扫描
                    if not self.stop_event.is_set():
                        await self.scan_once()
                    
        except asyncio.CancelledError:
            print("\n[取消] 扫描任务被取消")
            raise
        
        except Exception as e:
            print(f"\n[错误] 扫描出错: {e}")
            traceback.print_exc()
        
        finally:
            print("\n[扫描器] 正在清理资源...")
            # 这里可以添加清理代码
            print("[扫描器] 已停止\n")
    
    
    def stop(self):
        """停止扫描器"""
        print("[扫描器] 设置停止标志...")
        self.stop_event.set()


# 主程序
async def main():
    scanner = SimpleScanner(DB_CONFIG, SCRIPTS)
    
    # 用于强制退出的标志
    force_exit = False
    
    # 信号处理
    def signal_handler(sig, frame):
        nonlocal force_exit
        
        if force_exit:
            # 第二次 Ctrl+C，强制退出
            print("\n[强制] 强制终止程序！")
            sys.exit(1)
        
        print("\n\n[信号] 收到中断信号，正在优雅停止...")
        print("[提示] 再次按 Ctrl+C 可强制退出")
        scanner.stop()
        force_exit = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建运行任务
    run_task = asyncio.create_task(scanner.run())
    
    try:
        # 等待任务完成（带超时保护）
        await asyncio.wait_for(run_task, timeout=None)
    
    except asyncio.CancelledError:
        print("\n[取消] 主任务被取消")
    
    except KeyboardInterrupt:
        print("\n[中断] 收到键盘中断")
        scanner.stop()
        
        # 给5秒时间优雅退出
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            print("[超时] 强制取消任务...")
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    
    except Exception as e:
        print(f"\n[错误] 程序异常: {e}")
        traceback.print_exc()
    
    finally:
        print("="*70)
        print("程序已退出")
        print("="*70 + "\n")


if __name__ == "__main__":
    # 检查依赖
    required_packages = {
        'pymysql': 'pymysql',
        'psutil': 'psutil'
    }
    
    missing = []
    for package, install_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(install_name)
    
    if missing:
        print(f"缺少依赖包，请运行: pip install {' '.join(missing)}")
        exit(1)
    
    # 测试数据库连接
    print("="*70)
    print("测试数据库连接...")
    try:
        conn = pymysql.connect(**DB_CONFIG)
        print("连接成功")
        conn.close()
    except Exception as e:
        print(f"连接失败: {e}")
        exit(1)
    
    # 检查脚本文件
    print("\n检查脚本文件...")
    all_exist = True
    for key, config in SCRIPTS.items():
        path = config['script_path']
        if os.path.exists(path):
            print(f"{key}: {path}")
        else:
            print(f"{key}: {path} (不存在)")
            all_exist = False
    
    if not all_exist:
        print("\n警告: 部分脚本不存在，请检查配置")
    
    print("="*70)
    print()
    
    # 运行
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        print(f"\n程序异常退出: {e}")
        traceback.print_exc()