import os
import time
import cv2
import json
import logging
import threading
import base64
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np

from datetime import datetime
from threading import Thread
from queue import Queue
from collections import deque

from app.camera import Camera
from config import InitialConfig
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, Response
from skimage.metrics import structural_similarity as ssim


class BaseAnalyzer:
    def __init__(self):
        self.frame_queue = Queue(maxsize=10)
        self.change_queue = Queue(maxsize=5)
        self.processed_frames = Queue(maxsize=2)
        self.is_running = False
        self._status_lock = threading.Lock()
        self._lock = threading.Lock()
        self._threads = []
        self.output_dir = InitialConfig.OUTPUT_DIR
        self.max_saved_files = InitialConfig.MAX_SAVED_FILES
        self.analysis_enabled = False
        
    def start(self):
        """启动分析器"""
        if self.is_running:
            return
        
        try:
            self.is_running = True
            self._setup_logging()
            self._start_threads()
            self.logger.info(f"{self.__class__.__name__} started")
        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start {self.__class__.__name__}: {str(e)}")
            raise

    def stop(self):
        """停止分析器"""
        if not self.is_running:
            return
            
        self.is_running = False
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._clear_queues()
        self.logger.info(f"{self.__class__.__name__} stopped")

    def cleanup_old_files(self, extension):
        """清理旧文件"""
        try:
            files = [f for f in os.listdir(self.output_dir) if f.endswith(extension)]
            if len(files) > self.max_saved_files:
                files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
                for f in files[:(len(files) - self.max_saved_files)]:
                    os.remove(os.path.join(self.output_dir, f))
                    self.logger.info(f"Deleted old file: {f}")
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {str(e)}")

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
                    self.logger.info(f"Deleted old image: {f}")

        except Exception as e:
            self.logger.error(f"Error cleaning up old images: {str(e)}")

    def _clear_queues(self):
        """清空所有队列"""
        for queue in [self.frame_queue, self.change_queue, self.processed_frames]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    pass

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

            self.logger.info(f"Image saved successfully: {filepath}")

            return {
                'filepath': filepath,
                'base64_image': base64_image
            }

        except Exception as e:
            self.logger.error(f"Error saving image: {str(e)}")
            return None

    def llm_loop(self):
        """LLM处理循环"""
        def call_llm_service(image):
            try:
                # 清理旧图像
                self.cleanup_old_images()

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

    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger(self.__class__.__name__)
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

    def _start_threads(self):
        """启动分析线程"""
        self._threads = [
            Thread(target=self._analyze_loop),
            Thread(target=self.llm_loop)
        ]
        for thread in self._threads:
            thread.daemon = True
            thread.start()

    def _analyze_loop(self):
        """分析循环，子类需要实现"""
        raise NotImplementedError

    def _llm_analyze(self, data):
        """LLM分析方法"""
        try:
            # 清理旧文件
            extension = '.jpg' if isinstance(data, np.ndarray) else '.wav'
            self.cleanup_old_files(extension)

            # 保存数据
            if isinstance(data, np.ndarray):
                result = self.save_base64_image(data)
            else:
                result = self._save_audio_file(data)

            if result is None:
                self.logger.error("Failed to save data")
                return

            self.logger.info(f"Data saved: {result['filepath']}")
            # TODO: 实际的LLM服务调用代码

        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")

    def _save_audio_file(self, audio_data):
        """保存音频文件"""
        try:
            # 确保保存目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'audio_{timestamp}.wav'
            filepath = os.path.join(self.output_dir, filename)

            # 保存音频
            sf.write(filepath, audio_data, InitialConfig.AUDIO_SAMPLE_RATE)

            return {
                'filepath': filepath,
                'audio_data': audio_data
            }

        except Exception as e:
            self.logger.error(f"Error saving audio: {str(e)}")
            return None

    def toggle_analysis(self, enabled):
        """切换分析状态"""
        with self._status_lock:
            self.analysis_enabled = enabled
            self.logger.info(f"Analysis {'enabled' if enabled else 'disabled'}")
            return True


class VideoAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        # 添加 logger 初始化
        self.logger = logging.getLogger('VideoAnalyzer')
        # VideoAnalyzer 特有的属性
        self.last_frame = None
        self.fps = 0
        self.camera = Camera()
        self.video_source = None
        self._frame_times = deque(maxlen=30)
        self._last_frame_time = time.time()
        self.max_display_height = InitialConfig.MAX_DISPLAY_HEIGHT
        self.scale_percent = InitialConfig.SCALE_PERCENT

    def start(self, video_source=0):
        """启动视频分析器"""
        if self.is_running:
            return

        try:
            self.is_running = True
            self.video_source = video_source
            self.camera.start(video_source)

            # 启动分析线程
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
        """停止视频分析器"""
        if not self.is_running:
            return

        self.is_running = False
        self.camera.release()
        super().stop()

    def switch_camera(self, camera_index):
        """切换摄像头"""
        with self._lock:
            try:
                self.stop()
                time.sleep(1)
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

                        if not self.frame_queue.full() and self.analysis_enabled:
                            self.frame_queue.put(frame.copy())

            except Exception as e:
                self.logger.error(f"Error generating frames: {str(e)}")

            time.sleep(0.01)

    def _analyze_loop(self):
        """分析视频帧的循环"""
        def check_frame_change(frame1, frame2):
            try:
                if frame1 is None or frame2 is None:
                    return False

                if frame1.shape != frame2.shape:
                    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

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

                min_height = min(new_frame.shape[0] for new_frame in frames)
                resized_frames = []
                for new_frame in frames:
                    scale = min_height / new_frame.shape[0]
                    new_width = int(new_frame.shape[1] * scale)
                    resized = cv2.resize(new_frame, (new_width, min_height))
                    resized_frames.append(resized)

                return cv2.hconcat(resized_frames)

            except Exception as err:
                self.logger.error(f"Error processing change frames: {str(err)}")
                return None

        change_frames = []
        while self.is_running:
            try:
                if not self.frame_queue.empty() and self.analysis_enabled:
                    frame = self.frame_queue.get()
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

    def _preprocess_frame(self, frame):
        """预处理视频帧"""
        try:
            denoised = cv2.GaussianBlur(frame, (5, 5), 0)

            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            alpha = 1.2
            beta = 10
            adjusted = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

            return adjusted
        except Exception as e:
            self.logger.error(f"Frame preprocessing failed: {str(e)}")
            return frame

    def _resize_frame(self, frame):
        """等比缩放帧"""
        if frame is None:
            return None

        try:
            height, width = frame.shape[:2]
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
        while self.is_running:
            try:
                if not self.change_queue.empty():
                    concat_frame = self.change_queue.get()
                    self._llm_analyze(concat_frame)
            except Exception as e:
                self.logger.error(f"Error in LLM loop: {str(e)}")
            time.sleep(0.1)


class AudioAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.audio_queue = Queue(maxsize=10)
        self.sample_rate = 44100
        self.chunk_size = 1024 * 4
        self.threshold = 0.6
        self.last_audio = None
        self.stream = None
        
    def start(self):
        """启动音频分析器"""
        if self.is_running:
            return
            
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=self.chunk_size
            )
            self.stream.start()
            super().start()
        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start audio analyzer: {str(e)}")
            raise

    def stop(self):
        """停止音频分析器"""
        if not self.is_running:
            return

        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.stop()
            self.stream.close()
        super().stop()

    def _start_threads(self):
        """启动音频分析线程"""
        self._threads = [
            Thread(target=self._analyze_loop),
            Thread(target=self.llm_loop)
        ]
        for thread in self._threads:
            thread.daemon = True
            thread.start()
            
    def _audio_callback(self, indata, frames, time, status):
        """音频回调函数"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        if not self.audio_queue.full() and self.analysis_enabled:
            self.audio_queue.put(indata.copy())
            
    def _analyze_loop(self):
        """音频分析循环"""
        while self.is_running:
            try:
                if not self.audio_queue.empty() and self.analysis_enabled:
                    audio_data = self.audio_queue.get()
                    
                    if self.last_audio is not None:
                        if self._detect_audio_change(audio_data):
                            self.logger.info("Significant audio change detected")
                            self._save_audio_change(audio_data)
                            
                    self.last_audio = audio_data.copy()
                    
            except Exception as e:
                self.logger.error(f"Error in audio analysis: {str(e)}")
            time.sleep(0.01)
            
    def _detect_audio_change(self, audio_data):
        """检测音频变化"""
        if self.last_audio is None:
            return False
            
        try:
            # 计算音频特征
            mfcc1 = librosa.feature.mfcc(y=self.last_audio.flatten(), sr=self.sample_rate)
            mfcc2 = librosa.feature.mfcc(y=audio_data.flatten(), sr=self.sample_rate)
            
            # 计算相似度
            similarity = np.corrcoef(mfcc1.flatten(), mfcc2.flatten())[0, 1]
            return similarity < self.threshold
            
        except Exception as e:
            self.logger.error(f"Error detecting audio change: {str(e)}")
            return False
            
    def _save_audio_change(self, audio_data):
        """保存音频变化"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'audio_change_{timestamp}.wav'
            filepath = os.path.join(self.output_dir, filename)
            
            # 保存音频文件
            sf.write(filepath, audio_data, self.sample_rate)
            
            # 记录变化
            self._llm_analyze(audio_data)
            
        except Exception as e:
            self.logger.error(f"Error saving audio change: {str(e)}")


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
