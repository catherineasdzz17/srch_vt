import os
import time
import subprocess
import requests

class TmuxServiceManager:
    """tmux服务管理器 - 专门管理您的三窗口服务"""
    def __init__(self, session_name="video_services"):
        self.session_name = session_name
        self.main_window = "main"
        self.hunyuan_window = "hunyuan" 
        self.qwen_window = "qwen"
        self.service_port = 9090
        self.gpu_id = 0  # 根据你的GPU编号调整
    
    def send_tmux_command(self, window, command):
        """向指定tmux窗口发送命令"""
        try:
            cmd = f"tmux send-keys -t {self.session_name}:{window} '{command}' Enter"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            print(f"发送tmux命令失败: {e}")
            return False
    
    def send_ctrl_c(self, window):
        """向指定窗口发送Ctrl+C"""
        try:
            cmd = f"tmux send-keys -t {self.session_name}:{window} C-c"
            subprocess.run(cmd, shell=True, timeout=10)
            return True
        except Exception as e:
            print(f"发送Ctrl+C失败: {e}")
            return False
    
    def check_gpu_memory(self):
        """检查GPU显存使用情况"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > self.gpu_id:
                    used, total = map(float, lines[self.gpu_id].split(','))
                    usage_percent = (used / total) * 100
                    print(f"  [GPU{self.gpu_id}] 显存: {used:.0f}MB / {total:.0f}MB ({usage_percent:.1f}%)")
                    return used, total, usage_percent
            return 0, 0, 0
        except Exception as e:
            print(f"  [警告] 无法获取GPU显存信息: {e}")
            return 0, 0, 0
    
    def force_kill_service_processes(self):
        """强制杀死所有相关进程"""
        print("  强制清理残留进程...")
        
        # 杀死所有可能的python进程(根据端口或脚本名)
        kill_commands = [
            f"pkill -9 -f 'python.*api_main.py'",
            f"pkill -9 -f 'python.*{self.service_port}'",
            f"lsof -ti:{self.service_port} | xargs -r kill -9",
        ]
        
        for cmd in kill_commands:
            try:
                subprocess.run(cmd, shell=True, timeout=10, stderr=subprocess.DEVNULL)
            except:
                pass
        
        time.sleep(2)
    
    def clear_gpu_cache(self):
        """尝试清理GPU缓存"""
        print("  清理GPU缓存...")
        
        try:
            # 方法1: 使用Python清理
            clear_script = f"""
            python3 -c "
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                print('GPU cache cleared')
            "
            """
            subprocess.run(clear_script, shell=True, timeout=10, stderr=subprocess.DEVNULL)
            time.sleep(1)
        except:
            pass
        
        # 方法2: 如果有权限,尝试重置GPU(可选,可能需要root权限)
        # try:
        #     subprocess.run(
        #         f"nvidia-smi --gpu-reset -i {self.gpu_id}",
        #         shell=True, timeout=10, stderr=subprocess.DEVNULL
        #     )
        # except:
        #     pass
    
    def is_main_service_running(self):
        """检查main服务是否在运行"""
        try:
            response = requests.get(f'http://localhost:{self.service_port}/', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def stop_main_service(self):
        """停止main服务"""
        print("  停止main服务...")
        
        # 1. 发送Ctrl+C
        self.send_ctrl_c(self.main_window)
        time.sleep(3)
        
        # 2. 检查是否还在运行
        if self.is_main_service_running():
            print("  服务仍在运行,再次发送停止信号...")
            self.send_ctrl_c(self.main_window)
            time.sleep(3)
        
        # 3. 强制杀死进程
        self.force_kill_service_processes()
        
        # 4. 清理GPU缓存
        self.clear_gpu_cache()
        
        # 5. 最终检查
        if self.is_main_service_running():
            print("  [警告] 服务可能未完全停止")
        else:
            print("  main服务已完全停止")
    
    def start_main_service(self):
        """启动main服务"""
        print("  启动main服务...")
        
        # 检查显存状态
        used, total, usage = self.check_gpu_memory()
        
        # 激活conda环境并启动服务
        commands = [
            "export CUDA_VISIBLE_DEVICES=0",
            "conda activate main_env", 
            "python src/api_main.py"
        ]
        
        # 逐条发送命令
        for cmd in commands:
            if not self.send_tmux_command(self.main_window, cmd):
                print(f"  [错误] 发送命令失败: {cmd}")
                return False
            time.sleep(1)
        
        print("  服务启动命令已发送")
        return True
    
    def wait_for_service_ready(self, max_wait=120):
        """等待服务就绪"""
        print("  等待服务启动...")
        
        for i in range(max_wait):
            if self.is_main_service_running():
                print(f"  服务已就绪 (等待 {i+1} 秒)")
                # 服务启动后,再等待几秒确保完全就绪
                time.sleep(3)
                return True
            
            if (i + 1) % 10 == 0:  # 每10秒打印一次
                print(f"  等待服务启动... ({i+1}/{max_wait})")
            time.sleep(1)
        
        print("  [错误] 服务启动超时")
        return False
    
    def restart_main_service(self):
        """重启main服务 - 增强版"""
        print("\n" + "=" * 80)
        print("开始重启main服务(9090)...")
        print("=" * 80)
        
        # 1. 检查重启前的显存状态
        print("\n[重启前]")
        self.check_gpu_memory()
        
        # 2. 停止服务
        self.stop_main_service()
        
        # 3. 额外等待,确保完全停止
        print("\n  等待服务完全停止...")
        time.sleep(5)
        
        # 4. 检查停止后的显存状态
        print("\n[停止后]")
        self.check_gpu_memory()
        
        # 5. 启动服务
        if not self.start_main_service():
            print("\n[失败] 服务启动命令发送失败")
            return False
        
        # 6. 等待服务就绪
        if not self.wait_for_service_ready(max_wait=120):
            print("\n[失败] 服务启动超时")
            return False
        
        # 7. 检查启动后的显存状态
        print("\n[启动后]")
        self.check_gpu_memory()
        
        print("\n" + "=" * 80)
        print("main服务重启完成!")
        print("=" * 80 + "\n")
        
        return True
    
    def get_service_status(self):
        """获取所有服务状态"""
        used, total, usage = self.check_gpu_memory()
        
        status = {
            'main': self.is_main_service_running(),
            'session': self.check_tmux_session(),
            'port': self.service_port,
            'gpu_memory': {
                'used_mb': used,
                'total_mb': total,
                'usage_percent': usage
            }
        }
        return status
    
    def check_tmux_session(self):
        """检查tmux会话是否存在"""
        try:
            result = subprocess.run(
                f"tmux has-session -t {self.session_name}",
                shell=True, capture_output=True
            )
            return result.returncode == 0
        except:
            return False

    def print_status(self):
        """打印服务状态"""
        status = self.get_service_status()
        
        print("\n" + "=" * 80)
        print("服务状态")
        print("=" * 80)
        print(f"Tmux会话: {'✓ 运行中' if status['session'] else '✗ 未运行'}")
        print(f"Main服务: {'✓ 运行中' if status['main'] else '✗ 未运行'}")
        print(f"服务端口: {status['port']}")
        
        gpu_mem = status['gpu_memory']
        print(f"GPU显存: {gpu_mem['used_mb']:.0f}MB / {gpu_mem['total_mb']:.0f}MB ({gpu_mem['usage_percent']:.1f}%)")
        print("=" * 80 + "\n")


# 测试代码
if __name__ == "__main__":
    manager = TmuxServiceManager()
    
    # 打印当前状态
    manager.print_status()
    
    # 测试重启
    # if manager.check_tmux_session():
    #     print("测试重启服务...")
    #     success = manager.restart_main_service()
    #     print(f"重启{'成功' if success else '失败'}")
    #     
    #     # 再次打印状态
    #     manager.print_status()
    # else:
    #     print("错误: tmux会话不存在!")