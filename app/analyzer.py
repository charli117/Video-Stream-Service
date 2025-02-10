import os
import time
import cv2
import logging
import threading
import base64
import soundfile as sf
import numpy as np

from datetime import datetime
from threading import Thread
from queue import Queue
from collections import deque

from app.camera import Camera
from app.microphone import Microphone
from config import InitialConfig
from logging.handlers import RotatingFileHandler
from librosa import feature as librosa_feature
from skimage.metrics import structural_similarity as ssim


class BaseAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('BaseAnalyzer')
        self.frame_queue = Queue(maxsize=10)
        self.change_queue = Queue(maxsize=5)
        self.processed_frames = Queue(maxsize=2)
        self.is_running = False
        self.output_dir = InitialConfig.OUTPUT_DIR
        self.max_saved_files = InitialConfig.MAX_SAVED_FILES
        self.analysis_enabled = False
        self._status_lock = threading.Lock()
        self._lock = threading.Lock()
        self._threads = []

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
        """
        停止分析器
        1. 检查是否正在运行
        2. 将运行标志置为 False
        3. 等待所有线程结束并清理内部队列
        4. 记录停止日志
        """
        if not self.is_running:
            return

        self.is_running = False
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._clear_queues()
        self.logger.info(f"{self.__class__.__name__} stopped")

    def cleanup_old_files(self, extension, max_files=None):
        """
        清理旧文件
        根据文件扩展名检查输出目录中保存的文件数量，
        如果超过最大保留数量则按照文件修改时间排序删除最旧的文件。

        Args:
            extension (str): 文件扩展名，如 '.jpg' 或 '.wav'
            max_files (int|None): 最大允许保留文件数，默认为配置中的值
        """
        try:
            if max_files is None:
                max_files = (InitialConfig.MAX_SAVED_IMAGES
                             if extension == '.jpg'
                             else InitialConfig.MAX_SAVED_FILES)

            files = [f for f in os.listdir(self.output_dir) if f.endswith(extension)]
            if len(files) > max_files:
                files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
                for f in files[:(len(files) - max_files)]:
                    os.remove(os.path.join(self.output_dir, f))
                    self.logger.info(f"Deleted old file: {f}")
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {str(e)}")

    def toggle_analysis(self, enabled):
        """
        切换分析状态
        1. 记录当前状态和目标状态
        2. 若状态未改变则直接返回
        3. 改变状态、清理队列（当关闭分析时）及记录日志

        Args:
            enabled (bool): 目标状态，True 表示启动分析，False 表示关闭分析

        Returns:
            bool: 状态切换是否成功
        """
        with self._status_lock:
            try:
                self.logger.info(f"Toggling analysis to: {enabled}")  # 添加日志
                # 检查运行状态
                if not self.is_running:
                    self.logger.warning("Cannot toggle analysis when analyzer is not running")
                    return False

                # 检查状态是否需要改变
                if self.analysis_enabled == enabled:
                    self.logger.info(f"Analysis already {enabled}, no change needed")  # 添加日志
                    return True

                previous_state = self.analysis_enabled
                self.analysis_enabled = enabled

                # 清理队列
                if not enabled:
                    self._clear_queues()

                # 记录状态变化
                self.logger.info(f"Analysis state changed from {previous_state} to {enabled}")

                if enabled:
                    self.logger.info("Analysis started - monitoring for changes")
                else:
                    self.logger.info("Analysis stopped - no longer monitoring")

                return True

            except Exception as e:
                self.logger.error(f"Error toggling analysis: {str(e)}")
                return False

    def _clear_queues(self):
        """
        清空所有处理队列
        包括 frame_queue、change_queue 和 processed_frames 队列
        """
        for queue in [self.frame_queue, self.change_queue, self.processed_frames]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    pass

    def _save_base64_image(self, image):
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
            base64_image = base64.b64encode(buffer.tobytes()).decode('utf-8')

            self.logger.info(f"Image saved successfully: {filepath}")

            return {
                'filepath': filepath,
                'base64_image': base64_image
            }

        except Exception as e:
            self.logger.error(f"Error saving image: {str(e)}")
            return None

    def _save_audio_file(self, audio_data):
        """
        保存音频文件
        Args:
            audio_data: 音频数据信息

        Returns:
            dict: 包含保存文件信息的数据字典
        """
        try:
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'audio_{timestamp}.wav'
            filepath = os.path.join(self.output_dir, filename)

            # 确保音频数据格式正确
            if not isinstance(audio_data, np.ndarray):
                self.logger.error("Invalid audio data format")
                return None

            # 确保数据是float32类型
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # 规范化音频数据
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0

            # 确保数据形状正确
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            # 保存音频
            sf.write(
                filepath,
                audio_data,
                InitialConfig.AUDIO_SAMPLE_RATE,
                format='WAV',
                subtype='FLOAT'
            )

            return {
                'filepath': filepath,
                'audio_data': audio_data
            }

        except Exception as e:
            self.logger.error(f"Error saving audio: {str(e)}")
            return None

    # def llm_loop(self):
    #     """LLM处理循环"""
    #     def call_llm_service(image):
    #         try:
    #             # 清理旧图像
    #             self.cleanup_old_images()
    #
    #             # 保存图像并获取base64
    #             result = self.save_base64_image(image)
    #             if result is None and isinstance(result, dict):
    #                 self.logger.error("Failed to save image")
    #                 return
    #
    #             self.logger.info(f"Image saved: {result['filepath']}")
    #
    #             # TODO: 在这里添加实际的LLM服务调用代码
    #             # 例如：
    #             # response = llm_client.analyze_image(result['base64_image'])
    #             # return response
    #
    #         except Exception as err:
    #             self.logger.error(f"Error in LLM service: {str(err)}")
    #
    #     while self.is_running:
    #         try:
    #             if not self.change_queue.empty():
    #                 concat_frame = self.change_queue.get()
    #                 # 异步调用LLM服务
    #                 Thread(target=call_llm_service, args=(concat_frame,)).start()
    #         except Exception as e:
    #             self.logger.error(f"Error in LLM loop: {str(e)}")
    #
    #         time.sleep(0.1)

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

    def _thread_wrapper(self, target_func, thread_name):
        """线程包装器，用于异常处理和状态管理"""
        try:
            while self.is_running:
                target_func()
                time.sleep(0.01)  # 避免空循环占用CPU
        except Exception as e:
            self.logger.error(f"{thread_name} thread error: {str(e)}")
        finally:
            self.logger.info(f"{thread_name} thread stopped")

    def _start_threads(self):
        """启动分析线程"""
        self._threads = [
            Thread(target=self._thread_wrapper,
                   args=(self._analyze_loop, "analyze"),
                   name="AnalyzeThread"),
            Thread(target=self._thread_wrapper,
                   args=(self._llm_loop, "llm"),
                   name="LLMThread")
        ]

        for thread in self._threads:
            thread.daemon = True
            thread.start()
            self.logger.info(f"Started thread: {thread.name}")

    def _analyze_loop(self):
        """分析循环，子类需要实现"""
        raise NotImplementedError

    def _llm_analyze(self, data):
        """LLM分析方法"""
        try:
            result = None
            # 根据数据类型判断处理方式
            if isinstance(data, np.ndarray):
                if len(data.shape) == 3:  # 视频帧数据
                    extension = '.jpg'
                    self.cleanup_old_files(extension)
                    result = self._save_base64_image(data)
                else:  # 音频数据
                    extension = '.wav'
                    if data.size > 0:
                        self.cleanup_old_files(extension)
                        result = self._save_audio_file(data)

            if result:
                # TODO: 实际的LLM服务调用代码
                return result
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")

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


class VideoAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('VideoAnalyzer')
        self.camera = Camera()
        self.video_source = None
        self.last_frame = None
        self.fps = 0
        # 用于缓存检测到差异帧
        self._active_segment_frames = []
        self._frame_times = deque(maxlen=30)
        self._last_frame_time = time.time()
        self._max_display_height = InitialConfig.MAX_DISPLAY_HEIGHT
        self._blur_threshold = InitialConfig.BLUR_THRESHOLD
        self._similarity_threshold = InitialConfig.SIMILARITY_THRESHOLD
        self._change_frame_threshole = InitialConfig.CHANGE_FRAME_THRESHOLD

    def start(self, video_source=0):
        """启动视频分析器"""
        try:
            if self.is_running:
                return True

            self.video_source = video_source
            self.camera.start(video_source)

            # 确保摄像头正确初始化
            if not self.camera.is_initialized:
                raise RuntimeError("Camera failed to initialize")

            self.is_running = True
            self._setup_logging()
            self._start_threads()
            self.logger.info("VideoAnalyzer started successfully")
            return True

        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start VideoAnalyzer: {str(e)}")
            raise

    def stop(self):
        """停止视频分析器"""
        if not self.is_running:
            return

        try:
            self.is_running = False
            self.analysis_enabled = False  # 确保分析状态被重置

            # 在停止前，先刷新剩余的缓存帧
            self._flush_video_buffer()

            self.camera.release()

            # 等待所有线程结束
            for thread in self._threads:
                thread.join(timeout=1.0)

            self._clear_queues()

        except Exception as e:
            self.logger.error(f"Error stopping VideoAnalyzer: {str(e)}")

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
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_running:
            try:
                if not self.camera.is_initialized:
                    self.logger.error("Camera is not initialized")
                    time.sleep(1)
                    continue

                ret, frame = self.camera.read()
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Too many consecutive errors, stopping analyzer")
                        self.stop()
                        break
                    time.sleep(0.1)
                    continue

                consecutive_errors = 0  # 重置错误计数
                frame = self._resize_frame(frame)
                if frame is None:
                    self.logger.error("Failed to resize frame")
                    continue

                # 计算FPS
                current_time = time.time()
                self._frame_times.append(current_time - self._last_frame_time)
                self._last_frame_time = current_time
                self.fps = int(1.0 / (sum(self._frame_times) / len(self._frame_times)))

                # 压缩图像
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    self.logger.error("Failed to encode frame")
                    continue

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

        def is_blurry(frame_image):
            """
            使用拉普拉斯变换方法检测帧是否模糊。

            参数:
                frame_image: 输入帧。

            返回:
                bool: 如果帧是模糊的返回 True，否则返回 False。
            """
            gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var < self._blur_threshold, laplacian_var

        def check_frame_change(frame1, frame2):
            """
            使用 SSIM 检查两帧之间是否有显著变化，并过滤掉模糊帧。

            参数:
                frame1: 第一帧。
                frame2: 第二帧。
                blur_threshold: 过滤模糊帧的方差阈值。

            返回:
                bool: 如果有显著变化且帧不模糊返回 True，否则返回 False。
            """
            try:
                if frame1 is None or frame2 is None:
                    return False

                if frame1.shape != frame2.shape:
                    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

                # 计算第一帧和第二帧的模糊度
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                blur_flag1, blur1 = is_blurry(frame1)
                blur_flag2, blur2 = is_blurry(frame2)
                # 如果任一帧模糊直接返回
                if blur_flag1 or blur_flag2:
                    return False

                # 计算SSIM相似度
                ssim_score = ssim(gray1, gray2)
                # 如果符合变化条件，打印详细信息
                if ssim_score < self._similarity_threshold:
                    self.logger.info(f"检测到显著的视频帧变化: 上一帧模糊度={blur1:.2f}, 当前帧模糊度={blur2:.2f}, 相似度={ssim_score:.2f}")
                    return True
                else:
                    return False

            except Exception as err:
                self.logger.error(f"比较帧时出错: {str(err)}")
                return False

        while self.is_running:
            try:
                if not self.frame_queue.empty() and self.analysis_enabled:
                    frame = self.frame_queue.get()
                    processed_frame = self._preprocess_frame(frame)

                    if processed_frame is None:
                        continue

                    if processed_frame is not None and len(processed_frame.shape) == 3:
                        if self.last_frame is not None:
                            if check_frame_change(self.last_frame, processed_frame):
                                self._active_segment_frames.append(processed_frame.copy())
                                if len(self._active_segment_frames) >= self._change_frame_threshole:
                                    concat_frame = self._process_change_frames(self._active_segment_frames)
                                    if concat_frame is not None:
                                        if self.change_queue.full():
                                            try:
                                                self.change_queue.get_nowait()
                                            except Exception:
                                                pass
                                        self.change_queue.put(concat_frame)
                                        self.logger.info(f"已将 {len(self._active_segment_frames)} 帧拼接并入队")
                                    self._active_segment_frames = []

                        self.last_frame = processed_frame.copy()

            except Exception as e:
                self.logger.error(f"Error in analyze loop: {str(e)}")

            time.sleep(0.01)

    def _process_change_frames(self, frames):
        """
        将缓存的变化帧拼接为一个大帧。
        """
        try:
            if not frames:
                return None

            min_height = min(frame.shape[0] for frame in frames)
            resized_frames = []
            for frame in frames:
                scale = min_height / frame.shape[0]
                new_width = int(frame.shape[1] * scale)
                resized = cv2.resize(frame, (new_width, min_height))
                resized_frames.append(resized)

            return cv2.hconcat(resized_frames)

        except Exception as err:
            self.logger.error(f"Error processing change frames: {str(err)}")
            return None

    def _flush_video_buffer(self):
        """
        刷新当前缓存的视频变化帧，
        与音频类似：
        - 拼接当前缓存的帧
        - 将拼接结果放入 change_queue（注意队列满时需要释放一个元素）
        - 清空当前帧缓存
        """
        if self._active_segment_frames:
            concat_frame = self._process_change_frames(self._active_segment_frames)
            if concat_frame is not None:
                if self.change_queue.full():
                    try:
                        self.change_queue.get_nowait()
                    except Exception:
                        pass
                self.change_queue.put(concat_frame)
                self.logger.info(f"[flush_video_buffer] 保存了拼接视频帧，帧数: {len(self._active_segment_frames)}")
            self._active_segment_frames = []

    def _preprocess_frame(self, frame):
        """
        预处理视频帧，增强细节同时保持自然效果
        1. 轻微高斯模糊去噪
        2. 转换至LAB色彩空间，并只对光照通道应用CLAHE增强
        3. 合并通道并转换回BGR色彩空间
        4. 调整亮度和对比度

        Args:
            frame: 输入的视频帧数据

        Returns:
            处理后的帧数据
        """
        try:
            # 轻微的高斯模糊去噪，kernel size减小为(3,3)以保留更多细节
            denoised = cv2.GaussianBlur(frame, (3, 3), 0)

            # 转换到LAB色彩空间
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # 使用更温和的CLAHE参数
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # 合并通道
            enhanced = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # 轻微调整亮度和对比度
            alpha = 1.1  # 降低对比度增强
            beta = 5  # 降低亮度提升
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
            if height > self._max_display_height:
                scale = self._max_display_height / height
                new_width = int(width * scale)
                new_height = self._max_display_height
                frame = cv2.resize(frame, (new_width, new_height))
            return frame
        except Exception as e:
            self.logger.error(f"Error resizing frame: {str(e)}")
            return frame


class AudioAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('AudioAnalyzer')
        self.audio_queue = Queue(maxsize=10)
        self.last_audio = None
        self.is_stream_mode = InitialConfig.CAMERA_TYPE == 'stream'
        self.sample_rate = InitialConfig.AUDIO_SAMPLE_RATE
        self.chunk_size = InitialConfig.AUDIO_CHUNK_SIZE
        self.threshold = InitialConfig.AUDIO_CHANGE_THRESHOLD
        self.min_samples = int(InitialConfig.MIN_DURATION * self.sample_rate)
        self.microphone = Microphone()
        self.current_device = None  # 当前设备属性
        self._audio_thread = None  # 音频线程属性
        self._max_retries = 20
        # 用于缓存收到的音频数据
        self.active_segment = []
        self.accumulated_samples = 0
        self.retry_interval = 15
        self.retry_thread = None
        self.stopped = False

    def start(self, device_index=None):
        """启动音频分析器，并可接收设备索引参数"""
        if self.is_running:
            return

        try:
            # 设置设备索引
            if device_index is not None:
                self.current_device = device_index
            elif self.current_device is None and not self.is_stream_mode:
                available_devices = self.microphone.list_devices()
                if available_devices:
                    self.current_device = available_devices[0]['index']
                else:
                    raise RuntimeError("No audio devices available")

            # 释放旧资源
            if self.is_running:
                self.stop()
            self.microphone = Microphone()

            # 启动音频设备
            if not self.is_stream_mode:
                self.microphone.start(self.current_device)
            else:
                self.microphone.start()

            # 等待初始化，超时后更新错误状态而非直接抛出异常
            timeout = 5
            start_time = time.time()
            while not self.microphone.is_initialized and time.time() - start_time < timeout:
                time.sleep(0.1)
            if not self.microphone.is_initialized:
                self.audio_error = {"code": "INIT_FAILED", "message": "Audio设备初始化失败"}
                self.logger.error("Audio设备初始化失败")
                # 后台重试，不阻塞整个项目运行
                self._start_initialization_monitor()
                return

            self.is_running = True
            self._audio_thread = Thread(target=self.generate_audio, name="AudioCaptureThread")
            self._audio_thread.daemon = True
            self._audio_thread.start()
            self.logger.info(f"AudioAnalyzer started successfully with device {self.current_device}")

            super().start()

        except Exception:
            self.is_running = False
            # 同样启动后台重试机制
            self._start_initialization_monitor()
            return

    def _start_initialization_monitor(self):
        def monitor():
            while not self.stopped:
                if self.microphone.initialized:
                    logging.info("音频设备已初始化，停止重试。")
                    break
                logging.info("正在后台重试音频设备初始化...")
                self.microphone.start()  # 重试初始化
                time.sleep(self.retry_interval)
        self.retry_thread = threading.Thread(target=monitor, daemon=True)
        self.retry_thread.start()

    def switch_audio(self, device_index):
        """切换音频输入设备，利用完整停止流程重启音频分析器"""
        try:
            self.logger.info(f"切换音频设备到: {device_index}")

            # 记录当前分析状态
            was_analyzing = self.analysis_enabled

            # 停止音频分析器及相关线程，释放所有资产
            self.stop()

            # 创建新的 Microphone 实例
            self.microphone = Microphone()
            self.current_device = device_index

            # 重新启动音频分析器
            self.start(device_index)

            # 恢复之前的分析状态
            self.analysis_enabled = was_analyzing

            return True
        except Exception as e:
            self.logger.error(f"Error switching audio device: {str(e)}")
            return False

    def generate_audio(self):
        retry_count = 0
        output_stream = None

        # 初始化 error_state 防止后续使用错误
        self.error_state = None

        # 如果是在流模式下，初始化输出流，确保配置与输入一致
        # if self.microphone.is_stream_mode:
        #     try:
        #         import pyaudio
        #
        #         pa = pyaudio.PyAudio()
        #         output_stream = pa.open(
        #             format=pyaudio.paFloat32,
        #             channels=self.microphone.channels,
        #             rate=self.microphone.sample_rate,
        #             output=True,
        #             frames_per_buffer=self.microphone.chunk_size,
        #         )
        #         self.logger.info(f"Output stream started: {self.microphone.channels} channels @ {self.microphone.sample_rate}Hz")
        #     except Exception as e:
        #         self.logger.error(f"Error initializing output stream: {e}")

        while self.is_running:
            try:
                success, raw_data = self.microphone.read()
                if not success or raw_data is None:
                    retry_count += 1
                    if retry_count >= self._max_retries:
                        self.error_state = {
                            'code': 'AUDIO_READ_FAILED',
                            'message': '无法读取音频数据,请检查设备是否正常工作'
                        }
                        self.logger.error(f"[generate_audio] 连续 {self._max_retries}次 无法读取音频数据")
                        break
                    time.sleep(3)
                    continue

                # 重置错误计数
                retry_count = 0

                # 转换数据处理
                audio_data = self._convert_audio_data(raw_data)

                # 如果 self.last_audio 未初始化，则直接赋值，避免空处理
                if self.last_audio is None:
                    self.last_audio = audio_data.copy()
                    continue

                # 音频分析处理逻辑（振幅均值计算）：
                # similarity = np.abs(audio_data).mean()
                # if similarity > self.threshold and self.analysis_enabled:

                # 音频分析处理逻辑（MFCC 梅尔频率倒谱系数 计算），为MFCC计算动态设置n_fft，避免 n_fft 大于输入长度
                n_fft_value = min(2048, len(audio_data.flatten()))

                mfcc1 = librosa_feature.mfcc(y=self.last_audio.flatten(), sr=self.sample_rate, n_fft=n_fft_value)
                mfcc2 = librosa_feature.mfcc(y=audio_data.flatten(), sr=self.sample_rate, n_fft=n_fft_value)
                similarity = np.corrcoef(mfcc1.flatten(), mfcc2.flatten())[0, 1]

                # 如果相似度低于阈值且分析启用，则处理音频数据
                if similarity < self.threshold and self.analysis_enabled:
                    self.logger.info(f"检测到显著的音频流变化: 相识度 = {similarity}")
                    self.active_segment.append(audio_data)
                    self.accumulated_samples += audio_data.shape[0]

                # 更新 last_audio，后续对比使用当前数据
                self.last_audio = audio_data.copy()

                # 播放音频（仅在流模式下有效）
                # if output_stream is not None:
                #     try:
                #         output_stream.write(audio_data.tobytes())
                #     except Exception as e:
                #         self.logger.error(f"Error during audio playback: {e}")

            except Exception as e:
                self.logger.error(f"Error generating audio: {str(e)}")
                time.sleep(0.1)

        # 清理输出流资源
        # if output_stream is not None:
        #     output_stream.stop_stream()
        #     output_stream.close()

    def flush_audio_buffer(self):
        """
        刷新累积的音频缓冲区。
        - 拼接到目前为止收集的音频片段。
        - 确保合并后的音频满足最小持续时间要求。
        - 保存合并后的音频数据以供进一步分析。
        - 重置活动片段和累积样本。

        通常在停止音频分析器时调用此方法，以确保处理所有收集的音频数据。
        """
        if self.active_segment:
            # 拼接所有收集的音频片段
            combined = np.concatenate(self.active_segment, axis=0)

            # 保存合并后的音频数据
            save_result = self._llm_analyze(combined)
            if save_result:
                self.logger.info(f"[flush_audio_buffer] 保存了音频文件: {save_result['filepath']}")

            # 重置活动片段和累积样本
            self.active_segment = []
            self.accumulated_samples = 0

    def stop(self):
        """
        停止音频分析器
        1. 检查是否正在运行
        2. 设置关闭标志并等待音频线程结束
        3. 先flush缓存中的音频数据
        4. 释放音频资源后，重新创建 microphone 实例以便后续初始化
        5. 调用父类停止方法
        """
        if not self.is_running:
            return

        self.is_running = False
        if hasattr(self, '_audio_thread'):
            self._audio_thread.join(timeout=1.0)
        # flush剩余的音频数据
        self.flush_audio_buffer()
        # 释放音频资源后，重新创建 microphone 实例以便后续初始化
        # if self.microphone:
        #     self.microphone.release()
        #     self.microphone = Microphone()
        super().stop()
        self.stopped = True
        if self.retry_thread:
            self.retry_thread.join()

    def _analyze_loop(self):
        """
        音频数据分析循环
        循环检测分析是否开启及队列中是否有音频数据待处理
        每次从队列获取数据，保存音频文件，并启动异步 LLM 分析

        注意：当分析关闭时将暂停等待
        """
        while self.is_running:
            try:
                if not self.analysis_enabled:
                    time.sleep(0.05)
                    continue

                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()

                    # 保存音频文件
                    save_result = self._save_audio_file(audio_data)
                    if save_result:
                        self.logger.info(f"Successfully saved audio file: {save_result['filepath']}")

                    # 异步分析
                    Thread(target=self._llm_analyze, args=(audio_data,)).start()

            except Exception as e:
                self.logger.error(f"Error in audio analysis: {str(e)}")
            time.sleep(0.1)

    def _convert_audio_data(self, raw_data):
        """
        辅助函数：确保音频数据为 np.ndarray 且为 float32 类型，同时做形状调整和归一化
        """
        audio_data = raw_data if isinstance(raw_data, np.ndarray) else np.array(raw_data)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, self.microphone.channels)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0
        return audio_data
