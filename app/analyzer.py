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
import librosa.feature as librosa_feature
from skimage.metrics import structural_similarity as ssim


class BaseAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('BaseAnalyzer')
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

    def cleanup_old_files(self, extension, max_files=None):
        """
        清理旧文件
        Args:
            extension: 文件扩展名，如 '.jpg' 或 '.wav'
            max_files: 最大保留文件数，默认使用配置中的值
        """
        try:
            # 根据文件类型确定最大保留数量
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
        """切换分析状态"""
        with self._status_lock:
            try:
                # 检查运行状态
                if not self.is_running:
                    self.logger.warning("Cannot toggle analysis when analyzer is not running")
                    return False
                    
                # 检查状态是否需要改变
                if self.analysis_enabled == enabled:
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
                self.analysis_enabled = False  # 发生错误时重置状态
                return False

    def _clear_queues(self):
        """清空所有队列"""
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
        """保存音频文件"""
        try:
            # 确保保存目录存在
            os.makedirs(self.output_dir, exist_ok=True)

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
            # 根据数据类型判断处理方式
            if isinstance(data, np.ndarray):
                if len(data.shape) == 3:  # 视频帧数据
                    extension = '.jpg'
                    self.cleanup_old_files(extension)
                    result = self._save_base64_image(data)
                else:  # 音频数据
                    extension = '.wav'
                    self.logger.error("======")
                    if data.size > 0:
                        self.cleanup_old_files(extension)
                        result = self._save_audio_file(data)

                if result is None:
                    self.logger.error("Failed to save data")
                    return

                self.logger.info(f"Data saved: {result['filepath']}")
                # TODO: 实际的LLM服务调用代码

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
            self.camera.release()

            # 等待所有线程结束
            for thread in self._threads:
                thread.join(timeout=1.0)

            self._clear_queues()
            self.logger.info("VideoAnalyzer stopped successfully")

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
        while self.is_running:
            try:
                if not self.camera.is_initialized:
                    self.logger.error("Camera is not initialized")
                    time.sleep(1)
                    continue
                
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    self.logger.error("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
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

                                if len(change_frames) >= InitialConfig.CHANGE_FRAME_THRESHOLD:
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
        """预处理视频帧，增强细节但保持自然效果"""
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
            beta = 5    # 降低亮度提升
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


class AudioAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('AudioAnalyzer')
        # 增大队列容量
        self.audio_queue = Queue(maxsize=50)
        self.last_audio = None
        self.sample_rate = InitialConfig.AUDIO_SAMPLE_RATE
        self.chunk_size = InitialConfig.AUDIO_CHUNK_SIZE
        self.threshold = InitialConfig.AUDIO_CHANGE_THRESHOLD
        self.microphone = Microphone()
        self.current_device = None  # 当前设备属性
        self._audio_thread = None  # 音频线程属性

    def start(self, device_index=None):
        """启动音频分析器，并可接收设备索引参数"""
        if self.is_running:
            return

        try:
            self.logger.info("Starting audio analyzer thread...")
            # 如果传入设备索引，则使用传入值
            if (device_index is not None):
                self.current_device = device_index
            elif self.current_device is None:
                available_devices = self.microphone.list_devices()
                if available_devices:
                    self.current_device = available_devices[0]['index']
                else:
                    raise RuntimeError("No audio devices available")

            # 使用传入或选定的设备启动音频设备
            self.microphone.start(self.current_device)

            # 明确设置为正在运行
            self.is_running = True

            # 启动音频采集线程
            self._audio_thread = Thread(target=self.generate_audio, name="AudioCaptureThread")
            self._audio_thread.daemon = True
            self._audio_thread.start()
            self.logger.info("Audio capture thread started")

            # 启动父类线程管理（如果父类内部有额外处理）
            super().start()
            self.logger.info(f"AudioAnalyzer started successfully with device {self.current_device}")

        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start audio analyzer: {str(e)}")
            raise

    def switch_audio(self, device_index):
        """切换音频输入设备"""
        try:
            self.stop()
            time.sleep(0.5)
            self.microphone.start(device_index)
            self.current_device = device_index  # 更新当前设备索引
            return True
        except Exception as e:
            self.logger.error(f"Error switching audio device: {str(e)}")
            return False
            
    def generate_audio(self):
        """生成音频流，并进行静默检测和分割"""
        import numpy as np
        active_segment = []
        is_silent = True

        while self.is_running:
            try:
                success, raw_data = self.microphone.read()
                if not success or raw_data is None:
                    self.logger.warning("[generate_audio] 无法读取到音频数据")
                    time.sleep(0.01)
                    continue

                # 确保数据为 np.ndarray 且为 float32 类型
                if not isinstance(raw_data, np.ndarray):
                    audio_data = np.array(raw_data)
                    self.logger.info(f"[generate_audio] 转换 raw_data 至 numpy.ndarray, shape: {audio_data.shape}")
                else:
                    audio_data = raw_data

                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                if audio_data.max() > 1.0:
                    audio_data = audio_data / 32768.0

                amplitude = np.abs(audio_data).mean()

                if amplitude > InitialConfig.AUDIO_CHANGE_THRESHOLD:
                    active_segment.append(audio_data)
                    is_silent = False
                    self.logger.info("[generate_audio] 检测到非静默音频，添加到活动片段")
                else:
                    if not is_silent and active_segment:
                        complete_segment = np.concatenate(active_segment, axis=0)
                        self.logger.info("[generate_audio] 检测到静默音频，完成当前音频片段")
                        if self.analysis_enabled:
                            if self.audio_queue.qsize() < self.audio_queue.maxsize:
                                self.audio_queue.put_nowait(np.copy(complete_segment))
                                self.logger.info("[generate_audio] 音频片段已推送至队列")
                            else:
                                self.logger.warning("[generate_audio] 音频队列已满，跳过当前片段")
                        active_segment = []
                        is_silent = True
                    else:
                        self.logger.debug("[generate_audio] 静默状态，跳过当前数据")

            except Exception as e:
                self.logger.error(f"Error generating audio: {str(e)}")
            time.sleep(0.01)

    def stop(self):
        """停止音频分析器"""
        if not self.is_running:
            return

        self.is_running = False
        if hasattr(self, '_audio_thread'):
            self._audio_thread.join(timeout=1.0)
        self.microphone.release()
        super().stop()
            
    def _analyze_loop(self):
        """音频分析循环，从队列中获取音频片段并进行异步分析"""
        while self.is_running:
            try:
                if not self.analysis_enabled:
                    time.sleep(0.05)
                    continue

                # 增加对队列大小的监控
                if self.audio_queue.qsize() > self.audio_queue.maxsize * 0.8:
                    self.logger.warning(f"Audio queue is almost full: {self.audio_queue.qsize()}/{self.audio_queue.maxsize}, pausing audio capture")
                    time.sleep(1)  # 暂停音频采集
                    continue

                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # 异步调用LLM服务
                    Thread(target=self._llm_analyze, args=(audio_data,)).start()
                    self.logger.info(f"[analyze_loop] Audio analysis dispatched to LLM")

            except Exception as e:
                self.logger.error(f"Error in audio analysis: {str(e)}")
            time.sleep(0.1)
            
    # def _detect_audio_change(self, audio_data):
    #     """Detect audio changes using MFCC features"""
    #     if self.last_audio is None or not isinstance(audio_data, np.ndarray):
    #         return False
            
    #     try:
    #         # 确保音频数据是浮点型
    #         last_audio = self.last_audio.astype(np.float32).flatten()
    #         current_audio = audio_data.astype(np.float32).flatten()
            
    #         # 使用完整的导入路径计算音频特征
    #         mfcc1 = librosa_feature.mfcc(y=last_audio, sr=self.sample_rate)
    #         mfcc2 = librosa_feature.mfcc(y=current_audio, sr=self.sample_rate)
            
    #         # Calculate similarity
    #         similarity = np.corrcoef(mfcc1.flatten(), mfcc2.flatten())[0, 1]
            
    #         # Add detailed logging
    #         if similarity < self.threshold:
    #             self.logger.info(f"Audio change detected - Similarity: {similarity:.3f} (Threshold: {self.threshold})")
            
    #         return similarity < self.threshold
            
    #     except Exception as e:
    #         self.logger.error(f"Error in audio change detection: {str(e)}")
    #         return False

    def process_stream_audio(self, audio_data):
        """处理流式音频数据"""
        try:
            if self.analysis_enabled:
                # 直接调用 LLM 分析
                self._llm_analyze(audio_data)
                
        except Exception as e:
            self.logger.error(f"Error processing stream audio: {str(e)}")
            
    # def integrate_stream_audio_processing(self, stream_camera):
    #     """集成流式摄像头的音频处理"""
    #     try:
    #         if not isinstance(stream_camera, Camera):
    #             raise ValueError("Invalid stream camera instance")
    #
    #         # 设置音频回调函数
    #         stream_camera.set_audio_callback(self.process_stream_audio)
    #         self.logger.info("Integrated audio processing with StreamCamera")
    #
    #     except Exception as e:
    #         self.logger.error(f"Error integrating audio processing: {str(e)}")
