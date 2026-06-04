# diagnostic_tools.py
import torch
import psutil
import os
import sys
import functools
import traceback
from datetime import datetime

class MemoryMonitor:
    """内存和显存监控工具"""
    
    @staticmethod
    def get_memory_info():
        """获取当前内存和显存信息"""
        process = psutil.Process(os.getpid())
        
        info = {
            'cpu_memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_memory_percent': process.memory_percent(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            })
        
        return info
    
    @staticmethod
    def monitor(detailed=True):
        """装饰器：监控函数的内存使用"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__
                
                # 执行前
                before = MemoryMonitor.get_memory_info()
                print(f"\n{'='*60}")
                print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] 开始执行: {func_name}")
                print(f"{'='*60}")
                
                if detailed:
                    print(f"执行前:")
                    print(f"  💾 CPU内存: {before['cpu_memory_mb']:.2f} MB ({before['cpu_memory_percent']:.1f}%)")
                    if 'gpu_memory_allocated_mb' in before:
                        print(f"  🎮 GPU显存(已分配): {before['gpu_memory_allocated_mb']:.2f} MB")
                        print(f"  🎮 GPU显存(已保留): {before['gpu_memory_reserved_mb']:.2f} MB")
                
                # 执行函数
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    print(f"\n❌ 执行出错: {e}")
                    traceback.print_exc()
                    raise
                
                # 执行后
                after = MemoryMonitor.get_memory_info()
                
                cpu_diff = after['cpu_memory_mb'] - before['cpu_memory_mb']
                
                print(f"\n执行后:")
                print(f"  💾 CPU内存: {after['cpu_memory_mb']:.2f} MB ({after['cpu_memory_percent']:.1f}%)")
                
                if 'gpu_memory_allocated_mb' in after:
                    gpu_allocated_diff = after['gpu_memory_allocated_mb'] - before['gpu_memory_allocated_mb']
                    gpu_reserved_diff = after['gpu_memory_reserved_mb'] - before['gpu_memory_reserved_mb']
                    
                    print(f"  🎮 GPU显存(已分配): {after['gpu_memory_allocated_mb']:.2f} MB")
                    print(f"  🎮 GPU显存(已保留): {after['gpu_memory_reserved_mb']:.2f} MB")
                    
                    print(f"\n📈 增长:")
                    print(f"  💾 CPU内存: {'+' if cpu_diff >= 0 else ''}{cpu_diff:.2f} MB")
                    print(f"  🎮 GPU显存(已分配): {'+' if gpu_allocated_diff >= 0 else ''}{gpu_allocated_diff:.2f} MB")
                    print(f"  🎮 GPU显存(已保留): {'+' if gpu_reserved_diff >= 0 else ''}{gpu_reserved_diff:.2f} MB")
                    
                    # ⚠️ 警告：如果显存增长超过100MB
                    if gpu_allocated_diff > 100:
                        print(f"\n⚠️  警告: GPU显存增长过多! (+{gpu_allocated_diff:.2f} MB)")
                else:
                    print(f"\n📈 CPU内存增长: {'+' if cpu_diff >= 0 else ''}{cpu_diff:.2f} MB")
                
                print(f"{'='*60}\n")
                
                return result
            
            return wrapper
        return decorator


class ObjectSizeChecker:
    """检查对象占用内存大小"""
    
    @staticmethod
    def get_size(obj, seen=None):
        """递归计算对象大小（字节）"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        
        seen.add(obj_id)
        
        if isinstance(obj, dict):
            size += sum([ObjectSizeChecker.get_size(v, seen) for v in obj.values()])
            size += sum([ObjectSizeChecker.get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += ObjectSizeChecker.get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            try:
                size += sum([ObjectSizeChecker.get_size(i, seen) for i in obj])
            except:
                pass
        
        return size
    
    @staticmethod
    def format_size(size_bytes):
        """格式化字节大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    @staticmethod
    def analyze_dict(data: dict, top_n=10):
        """分析字典中各个键的大小"""
        print(f"\n{'='*60}")
        print(f"📦 分析字典内容大小")
        print(f"{'='*60}")
        
        sizes = {}
        for key, value in data.items():
            sizes[key] = ObjectSizeChecker.get_size(value)
        
        # 按大小排序
        sorted_items = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
        
        total_size = sum(sizes.values())
        print(f"总大小: {ObjectSizeChecker.format_size(total_size)}\n")
        
        print(f"前 {top_n} 个最大的键:")
        for i, (key, size) in enumerate(sorted_items[:top_n], 1):
            percent = (size / total_size * 100) if total_size > 0 else 0
            print(f"  {i}. {key}: {ObjectSizeChecker.format_size(size)} ({percent:.1f}%)")
            
            # ⚠️ 如果某个键超过50MB，给出警告
            if size > 50 * 1024 * 1024:
                print(f"      ⚠️  这个键占用了大量内存!")
        
        print(f"{'='*60}\n")
        
        return sorted_items


class TorchDiagnostics:
    """PyTorch 相关诊断工具"""
    
    @staticmethod
    def check_cuda_context():
        """检查 CUDA 上下文"""
        if not torch.cuda.is_available():
            print("❌ CUDA 不可用")
            return
        
        print(f"\n{'='*60}")
        print(f"🎮 CUDA 诊断")
        print(f"{'='*60}")
        
        print(f"设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")
        
        # 显存信息
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"\n显存使用:")
        print(f"  已分配: {allocated:.2f} MB")
        print(f"  已保留: {reserved:.2f} MB")
        print(f"  峰值分配: {max_allocated:.2f} MB")
        print(f"  缓存(未使用): {reserved - allocated:.2f} MB")
        
        # ⚠️ 如果缓存过多，给出警告
        if (reserved - allocated) > 500:
            print(f"\n⚠️  警告: 有大量显存被缓存未使用 ({reserved - allocated:.2f} MB)")
            print(f"   建议执行: torch.cuda.empty_cache()")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def check_no_grad_context():
        """检查是否在 no_grad 上下文中"""
        is_enabled = torch.is_grad_enabled()
        print(f"\n{'='*60}")
        print(f"🔍 梯度检查")
        print(f"{'='*60}")
        print(f"梯度计算: {'启用 ⚠️' if is_enabled else '禁用 ✅'}")
        
        if is_enabled:
            print(f"\n⚠️  警告: 当前在推理模式，但梯度计算是启用的！")
            print(f"   这会占用额外显存来存储计算图")
            print(f"   建议在函数上添加装饰器: @torch.no_grad()")
            print(f"   或使用: with torch.no_grad():")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def find_tensors_in_object(obj, path="root", max_depth=5, current_depth=0):
        """递归查找对象中的 Tensor"""
        if current_depth > max_depth:
            return []
        
        tensors = []
        
        if isinstance(obj, torch.Tensor):
            device = obj.device
            size_mb = obj.element_size() * obj.nelement() / 1024 / 1024
            tensors.append({
                'path': path,
                'shape': tuple(obj.shape),
                'dtype': str(obj.dtype),
                'device': str(device),
                'size_mb': size_mb,
                'requires_grad': obj.requires_grad
            })
        
        elif isinstance(obj, dict):
            for key, value in obj.items():
                tensors.extend(
                    TorchDiagnostics.find_tensors_in_object(
                        value, f"{path}['{key}']", max_depth, current_depth + 1
                    )
                )
        
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                tensors.extend(
                    TorchDiagnostics.find_tensors_in_object(
                        item, f"{path}[{i}]", max_depth, current_depth + 1
                    )
                )
        
        elif hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                tensors.extend(
                    TorchDiagnostics.find_tensors_in_object(
                        value, f"{path}.{key}", max_depth, current_depth + 1
                    )
                )
        
        return tensors
    
    @staticmethod
    def analyze_result_tensors(result):
        """分析结果中的 Tensor"""
        print(f"\n{'='*60}")
        print(f"🔍 查找结果中的 Tensor")
        print(f"{'='*60}")
        
        tensors = TorchDiagnostics.find_tensors_in_object(result)
        
        if not tensors:
            print("✅ 未发现 Tensor (这是好事!)")
        else:
            print(f"⚠️  发现 {len(tensors)} 个 Tensor:\n")
            
            total_gpu_mb = 0
            total_cpu_mb = 0
            
            for i, tensor_info in enumerate(tensors, 1):
                print(f"{i}. {tensor_info['path']}")
                print(f"   形状: {tensor_info['shape']}")
                print(f"   类型: {tensor_info['dtype']}")
                print(f"   设备: {tensor_info['device']}")
                print(f"   大小: {tensor_info['size_mb']:.2f} MB")
                print(f"   需要梯度: {tensor_info['requires_grad']}")
                
                if 'cuda' in tensor_info['device']:
                    total_gpu_mb += tensor_info['size_mb']
                    print(f"   ⚠️  这个 Tensor 在 GPU 上!")
                else:
                    total_cpu_mb += tensor_info['size_mb']
                
                print()
            
            if total_gpu_mb > 0:
                print(f"⚠️⚠️⚠️  GPU 上的 Tensor 总计: {total_gpu_mb:.2f} MB")
                print(f"   这些 Tensor 会持续占用显存，直到被删除!")
                print(f"   建议: 将 Tensor 转换为 numpy 或 list")
            
            if total_cpu_mb > 0:
                print(f"CPU 上的 Tensor 总计: {total_cpu_mb:.2f} MB")
        
        print(f"{'='*60}\n")
        
        return tensors


print("✅ 诊断工具已加载")