import os
import time
import cv2
import json
import logging
import threading
import base64

from datetime import datetime
from threading import Thread
from queue import Queue
from collections import deque
from app.camera import Camera
from config import InitialConfig
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, Response
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger('utils')


class VideoAnalyzer:
    def __init__(self):
        self.frame_queue = Queue(maxsize=10)
        self.change_queue = Queue(maxsize=5)
        self.processed_frames = Queue(maxsize=2)
        self.last_frame = None
        self.is_running = False
        self.fps = 0
        self.camera = Camera()
        self.video_source = None
        self.analysis_enabled = False
        self._lock = threading.Lock()
        self._setup_logging()
        self._threads = []
        self._frame_times = deque(maxlen=30)
        self._last_frame_time = time.time()
        self.max_saved_images = InitialConfig.MAX_SAVED_IMAGES
        self._status_lock = threading.Lock()  # 添加状态锁
        self.max_display_height = InitialConfig.MAX_DISPLAY_HEIGHT  # 最大显示高度，用于等比缩放
        self.scale_percent = InitialConfig.SCALE_PERCENT  # JPEG压缩质量
        self.output_dir = InitialConfig.OUTPUT_DIR

    # def update_fps(self):
    #     """更新FPS计算"""
    #     with self._status_lock:
    #         current_time = time.time()
    #         self._frame_times.append(current_time - self._last_frame_time)
    #         self._last_frame_time = current_time
    #         if len(self._frame_times) >= 2:  # 至少需要两个时间点
    #             self.fps = int(1.0 / (sum(self._frame_times) / len(self._frame_times)))

    def start(self, video_source=0):
        """启动分析器"""
        if self.is_running:
            return

        try:
            self.is_running = True
            self.video_source = video_source

            # 初始化摄像头
            self.camera.start(video_source)

            # 只启动分析线程
            self._threads = [
                Thread(target=self._analyze_loop),
                Thread(target=self._llm_loop)
            ]

            for thread in self._threads:
                thread.daemon = True
                thread.start()

            self.logger.info("VideoAnalyzer started")

        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start VideoAnalyzer: {str(e)}")
            raise

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False
        self.camera.release()

        # 等待所有线程结束
        for thread in self._threads:
            thread.join(timeout=1.0)

        # 清空队列
        self._clear_queues()
        self.logger.info("VideoAnalyzer stopped")

    def get_status(self):
        """获取当前状态（线程安全）"""
        with self._status_lock:
            return {
                'is_running': self.is_running,
                'current_camera': self.video_source,
                'fps': self.fps,
                'camera_info': self.camera.get_info() if self.camera else {},
                'analysis_enabled': self.analysis_enabled  # 添加分析状态
            }

    def switch_camera(self, camera_index):
        """切换摄像头"""
        with self._lock:
            try:
                # 停止当前摄像头
                self.stop()
                time.sleep(1)  # 等待资源释放

                # 启动新摄像头
                self.start(camera_index)
                return True
            except Exception as e:
                self.logger.error(f"Error switching camera: {str(e)}")
                return False

    def generate_frames(self):
        """生成视频流"""
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    # 等比缩放
                    frame = self._resize_frame(frame)

                    # 计算FPS
                    current_time = time.time()
                    self._frame_times.append(current_time - self._last_frame_time)
                    self._last_frame_time = current_time
                    self.fps = int(1.0 / (sum(self._frame_times) / len(self._frame_times)))

                    # 压缩图像
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' +
                               buffer.tobytes() + b'\r\n')

                        # 将原始帧放入队列供分析使用
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame.copy())

            except Exception as e:
                self.logger.error(f"Error generating frames: {str(e)}")

            time.sleep(0.01)

    def save_base64_image(self, image):
        """
        将图像保存为文件

        Args:
            image: numpy数组格式的图像

        Returns:
            str: 保存的文件路径
        """
        try:
            # 确保保存目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            # 生成文件名（使用时间戳）
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'frame_{timestamp}.jpg'
            filepath = os.path.join(self.output_dir, filename)

            # 保存图像
            cv2.imwrite(filepath, image)

            # 转换为base64
            _, buffer = cv2.imencode('.jpg', image)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            logger.info(f"Image saved successfully: {filepath}")

            return {
                'filepath': filepath,
                'base64_image': base64_image
            }

        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return None

    def cleanup_old_images(self, max_files=InitialConfig.MAX_SAVED_IMAGES):
        """清理旧图像文件"""
        try:
            # 获取目录中的所有图像文件
            files = [f for f in os.listdir(self.output_dir) if f.endswith('.jpg')]

            # 如果文件数量超过最大值，删除最旧的文件
            if len(files) > max_files:
                # 按修改时间排序
                files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))

                # 删除多余的文件
                for f in files[:(len(files) - max_files)]:
                    os.remove(os.path.join(self.output_dir, f))
                    logger.info(f"Deleted old image: {f}")

        except Exception as e:
            logger.error(f"Error cleaning up old images: {str(e)}")

    def _clear_queues(self):
        """清空所有队列"""
        for queue in [self.frame_queue, self.change_queue, self.processed_frames]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    pass

    def _setup_logging(self):
        self.logger = logging.getLogger('VideoAnalyzer')
        self.logger.setLevel(logging.INFO)

        # 文件处理器(带轮转)
        file_handler = RotatingFileHandler(
            InitialConfig.LOG_PATH,
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(InitialConfig.LOG_FORMAT))
        self.logger.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(InitialConfig.LOG_FORMAT))
        self.logger.addHandler(console_handler)

    def _preprocess_frame(self, frame):
        try:
            # 高斯去噪
            denoised = cv2.GaussianBlur(frame, (5, 5), 0)

            # 自适应直方图均衡化
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # 调整亮度和对比度
            alpha = 1.2  # 对比度
            beta = 10  # 亮度
            adjusted = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

            return adjusted
        except Exception as e:
            self.logger.error(f"Frame preprocessing failed: {str(e)}")
            return frame

    # def _capture_loop(self):
    #     """优化的捕获循环"""
    #     while self.is_running:
    #         try:
    #             ret, frame = self.camera.read()
    #             if ret:
    #                 # 计算FPS
    #                 current_time = time.time()
    #                 self._frame_times.append(current_time - self._last_frame_time)
    #                 self._last_frame_time = current_time
    #                 self.fps = int(1.0 / (sum(self._frame_times) / len(self._frame_times)))
    #
    #                 # 如果队列满了，移除旧帧
    #                 if self.frame_queue.full():
    #                     try:
    #                         self.frame_queue.get_nowait()
    #                     except:
    #                         pass
    #
    #                 # 压缩图像以减少数据量
    #                 compressed_frame = self._compress_frame(frame)
    #                 self.frame_queue.put(compressed_frame, timeout=0.1)
    #
    #         except Exception as e:
    #             self.logger.error(f"Error in capture loop: {str(e)}")
    #             time.sleep(0.01)

    def toggle_analysis(self, enabled):
        """切换分析状态"""
        with self._status_lock:
            self.analysis_enabled = enabled
            self.logger.info(f"Analysis {'enabled' if enabled else 'disabled'}")
            return True

    def _analyze_loop(self):
        """分析视频帧的循环"""
        def check_frame_change(frame1, frame2):
            try:
                if frame1 is None or frame2 is None:
                    return False

                # 确保帧大小一致
                if frame1.shape != frame2.shape:
                    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

                # 转换为灰度图
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                # 计算结构相似性
                score = ssim(gray1, gray2)
                return score < 0.8

            except Exception as err:
                self.logger.error(f"Error comparing frames: {str(err)}")
                return False

        def process_change_frames(frames):
            """处理变化帧"""
            try:
                if not frames:
                    return None

                # 找到所有帧中的最小高度
                min_height = min(new_frame.shape[0] for new_frame in frames)

                # 等比缩放所有帧
                resized_frames = []
                for new_frame in frames:
                    # 计算缩放比例
                    scale = min_height / new_frame.shape[0]
                    new_width = int(new_frame.shape[1] * scale)
                    resized = cv2.resize(new_frame, (new_width, min_height))
                    resized_frames.append(resized)

                # 水平拼接
                return cv2.hconcat(resized_frames)

            except Exception as err:
                self.logger.error(f"Error processing change frames: {str(err)}")
                return None

        change_frames = []
        while self.is_running:
            try:
                if not self.frame_queue.empty() and self.analysis_enabled:
                    frame = self.frame_queue.get()

                    # 预处理
                    processed_frame = self._preprocess_frame(frame)

                    if processed_frame is not None and len(processed_frame.shape) == 3:
                        if self.last_frame is not None:
                            if check_frame_change(self.last_frame, processed_frame):
                                self.logger.info("Significant frame change detected")
                                change_frames.append(processed_frame.copy())

                                if len(change_frames) >= 5:
                                    concat_frame = process_change_frames(change_frames)
                                    if concat_frame is not None:
                                        if self.change_queue.full():
                                            try:
                                                self.change_queue.get_nowait()
                                            except:
                                                pass
                                        self.change_queue.put(concat_frame)
                                    change_frames = []

                        self.last_frame = processed_frame.copy()

            except Exception as e:
                self.logger.error(f"Error in analyze loop: {str(e)}")

            time.sleep(0.01)

    def _compress_frame(self, frame):
        """压缩图像"""
        try:
            # 调整图像大小
            width = int(frame.shape[1] * self.scale_percent / 100)
            height = int(frame.shape[0] * self.scale_percent / 100)
            frame = cv2.resize(frame, (width, height))

            # JPEG压缩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # 85%质量
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            return buffer

        except Exception as e:
            self.logger.error(f"Error compressing frame: {str(e)}")
            return frame

    def _resize_frame(self, frame):
        """等比缩放帧"""
        if frame is None:
            return None

        try:
            height, width = frame.shape[:2]
            # 只有当高度超过最大显示高度时才进行缩放
            if height > self.max_display_height:
                scale = self.max_display_height / height
                new_width = int(width * scale)
                new_height = self.max_display_height
                frame = cv2.resize(frame, (new_width, new_height))
            return frame
        except Exception as e:
            self.logger.error(f"Error resizing frame: {str(e)}")
            return frame

    def _llm_loop(self):
        """LLM处理循环"""
        def call_llm_service(image):
            try:
                # 清理旧图像
                self.cleanup_old_images(self.max_saved_images)

                # 保存图像并获取base64
                result = self.save_base64_image(image)
                if result is None and isinstance(result, dict):
                    self.logger.error("Failed to save image")
                    return

                self.logger.info(f"Image saved: {result['filepath']}")

                # TODO: 在这里添加实际的LLM服务调用代码
                # 例如：
                # response = llm_client.analyze_image(result['base64_image'])
                # return response

            except Exception as err:
                self.logger.error(f"Error in LLM service: {str(err)}")

        while self.is_running:
            try:
                if not self.change_queue.empty():
                    concat_frame = self.change_queue.get()
                    # 异步调用LLM服务
                    Thread(target=call_llm_service, args=(concat_frame,)).start()
            except Exception as e:
                self.logger.error(f"Error in LLM loop: {str(e)}")

            time.sleep(0.1)


class VideoWeb:
    def __init__(self, analyzer):
        self.app = Flask(__name__)
        self.analyzer = analyzer

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/stats')
        def stats():
            return json.dumps({
                'fps': self.analyzer.fps,
                'queue_size': self.analyzer.frame_queue.qsize(),
                'running': self.analyzer.is_running,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    def generate_frames(self):
        while True:
            if not self.analyzer.processed_frames.empty():
                frame = self.analyzer.processed_frames.get()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)

    def run(self, host='0.0.0.0', port=5008):
        self.app.run(host=host, port=port, threaded=True)
