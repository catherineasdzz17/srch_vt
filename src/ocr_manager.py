import threading
import time
import gc
import logging

logger = logging.getLogger(__name__)


class OCRManager:
    """OCR实例管理器,自动重建实例防止显存泄漏"""
    def __init__(self, ocr_instance=None, max_videos_per_instance=2):
        self.ocr = ocr_instance
        self.processed_count = 0
        self.max_videos_per_instance = max_videos_per_instance
        self.lock = threading.Lock()  # 线程安全
        logger.info(f"OCR管理器初始化 (每{max_videos_per_instance}个视频重建)")
    

    def get_ocr(self):
        """获取OCR实例，必要时重建"""
        with self.lock:
            # 检查是否需要重建
            if self.ocr is None:
                logger.info("OCR实例不存在，创建新实例...")
                self._rebuild_ocr()
            elif self.processed_count >= self.max_videos_per_instance:
                logger.info(f"已处理{self.processed_count}个视频，达到阈值，重建OCR实例...")
                self._rebuild_ocr()
            
            return self.ocr
    
    
    def _rebuild_ocr(self):
        """重建OCR实例"""
        try:
            # 1. 清理旧实例
            if self.ocr is not None:
                logger.info("清理旧OCR实例...")
                
                # 安全清理OCR实例本身
                self._safe_cleanup_ocr_instance()
                
                # 删除引用
                del self.ocr
                self.ocr = None
            
            # 2. 只清理本进程的垃圾，不影响GPU全局
            gc.collect()
            
            # 3. 等待资源释放
            time.sleep(2)
            
            # 4. 创建新实例
            logger.info("创建新OCR实例...")
            from paddleocr import PaddleOCR
            
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                show_log=False,
                use_gpu=True,
                gpu_mem=500,  # 限制显存使用（单位MB）
                enable_mkldnn=False,
                use_tensorrt=False,
            )
            
            # 5. 重置计数器
            self.processed_count = 0
            
            # 6. 验证实例创建成功
            if self.ocr is None:
                raise RuntimeError("OCR实例创建失败")
            
            logger.info("OCR实例重建成功")
            
        except Exception as e:
            logger.error(f"OCR实例重建失败: {e}")
            self.ocr = None
            raise
    
    
    def _safe_cleanup_ocr_instance(self):
        """
        安全清理OCR实例，只清理当前实例的资源
        不影响同GPU上的其他进程
        """
        try:
            logger.info("开始安全清理OCR实例...")
            
            # 1. 尝试调用OCR实例的内部清理方法
            if hasattr(self.ocr, 'text_detector'):
                try:
                    # 清理检测器
                    if hasattr(self.ocr.text_detector, 'predictor'):
                        del self.ocr.text_detector.predictor
                    del self.ocr.text_detector
                    logger.debug("检测器已清理")
                except Exception as e:
                    logger.warning(f"清理检测器失败: {e}")
            
            if hasattr(self.ocr, 'text_recognizer'):
                try:
                    # 清理识别器
                    if hasattr(self.ocr.text_recognizer, 'predictor'):
                        del self.ocr.text_recognizer.predictor
                    del self.ocr.text_recognizer
                    logger.debug("识别器已清理")
                except Exception as e:
                    logger.warning(f"清理识别器失败: {e}")
            
            if hasattr(self.ocr, 'text_classifier'):
                try:
                    # 清理分类器
                    if hasattr(self.ocr.text_classifier, 'predictor'):
                        del self.ocr.text_classifier.predictor
                    del self.ocr.text_classifier
                    logger.debug("分类器已清理")
                except Exception as e:
                    logger.warning(f"清理分类器失败: {e}")
            
            # 2. 清理可能存在的其他属性
            if hasattr(self.ocr, 'use_gpu'):
                self.ocr.use_gpu = False
            
            logger.info("OCR实例内部资源清理完成")
            
        except Exception as e:
            logger.warning(f"OCR实例清理过程出现异常: {e}")
    

    def mark_video_processed(self):
        """标记已处理一个视频"""
        with self.lock:
            self.processed_count += 1
            logger.info(f"OCR已处理视频数: {self.processed_count}/{self.max_videos_per_instance}")
    

    def force_rebuild(self):
        """强制重建OCR实例"""
        with self.lock:
            logger.warning("强制重建OCR实例")
            self.processed_count = self.max_videos_per_instance  # 触发重建
            self._rebuild_ocr()
    
    
    def __del__(self):
        """析构函数，确保资源释放"""
        try:
            if self.ocr is not None:
                logger.info("OCRManager析构，清理资源...")
                self._safe_cleanup_ocr_instance()
                del self.ocr
                gc.collect()
        except Exception as e:
            logger.warning(f"OCRManager析构失败: {e}")