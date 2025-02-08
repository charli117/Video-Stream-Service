import cv2
import time
import logging
import numpy as np
import requests
import av
import threading
from typing import Optional
from config import InitialConfig


class Camera:
    def __init__(self):
        self.logger = logging.getLogger('Camera')
        self.camera = None
        self.is_initialized = False  # 添加初始化标志
        # 添加基本属性
        self.width = 0
        self.height = 0
        self.fps = 0
        self._audio_callback = None  # 添加音频回调属性

    def start(self, source=0):
        try:
            if InitialConfig.CAMERA_TYPE == 'local':
                self.camera = LocalCamera()
                self.camera.start(source)
            elif InitialConfig.CAMERA_TYPE == 'stream':
                self.camera = StreamCamera()
                self.camera.start(source)
            else:
                raise ValueError("Invalid CAMERA_TYPE in configuration")

            self.is_initialized = self.camera.is_initialized
            self.width = self.camera.width
            self.height = self.camera.height
            self.fps = self.camera.fps
            self.logger.info("Camera started successfully")
        except Exception as e:
            self.logger.error(f"Error starting camera: {str(e)}")
            raise RuntimeError(f"Failed to start camera: {str(e)}")

    def get_info(self):
        """获取摄像头信息"""
        if self.camera is None:
            return {
                'initialized': False,
                'width': 0,
                'height': 0,
                'fps': 0
            }
        return self.camera.get_info()

    def read(self):
        if self.camera is None:
            return False, None
        return self.camera.read()

    def release(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    @staticmethod
    def list_cameras():
        if InitialConfig.CAMERA_TYPE == 'local':
            return LocalCamera.list_cameras()
        elif InitialConfig.CAMERA_TYPE == 'stream':
            return StreamCamera.list_cameras()
        else:
            return []

    @staticmethod
    def is_valid_camera(index):
        if InitialConfig.CAMERA_TYPE == 'local':
            return LocalCamera.is_valid_camera(index)
        elif InitialConfig.CAMERA_TYPE == 'stream':
            return StreamCamera.is_valid_camera(index)
        else:
            return False

    @staticmethod
    def get_device_name(index):
        if InitialConfig.CAMERA_TYPE == 'local':
            return LocalCamera.get_device_name(index)
        elif InitialConfig.CAMERA_TYPE == 'stream':
            return StreamCamera.get_device_name(index)
        else:
            return f'Unknown Camera {index}'

    @staticmethod
    def update_device_names(device_names):
        if InitialConfig.CAMERA_TYPE == 'local':
            LocalCamera.update_device_names(device_names)
        elif InitialConfig.CAMERA_TYPE == 'stream':
            StreamCamera.update_device_names(device_names)
        else:
            raise ValueError("Invalid CAMERA_TYPE in configuration")


class LocalCamera:
    _device_names = {}  # 类变量，用于存储设备名称映射

    def __init__(self):
        self.logger = logging.getLogger('LocalCamera')  # 添加 logger 初始化
        self.cap = None
        self.width = None
        self.height = None
        self.fps = InitialConfig.DEFAULT_FPS
        self.is_initialized = False
        self._buffer_size = InitialConfig.MAX_QUEUE_SIZE

    @staticmethod
    def is_valid_camera(index):
        """公共方法：检查摄像头是否有效"""
        return LocalCamera._is_valid_camera(index)

    @classmethod
    def get_device_name(cls, index):
        """公共方法：获取设备名称"""
        return cls._device_names.get(str(index), f'Camera {index}')

    @staticmethod
    def _is_valid_camera(index):
        """检查摄像头是否有效"""
        try:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        except:
            return False

    @classmethod
    def update_device_names(cls, device_names):
        """更新设备名称映射"""
        cls._device_names = device_names

    @staticmethod
    def list_cameras():
        """列出所有可用摄像头"""
        available_cameras = []
        logger = logging.getLogger('Camera')

        for i in range(4):
            if LocalCamera._is_valid_camera(i):
                cap = cv2.VideoCapture(i)
                info = {
                    'index': i,
                    'name': LocalCamera._device_names.get(str(i), f'Camera {i}'),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                }
                cap.release()
                available_cameras.append(info)
                logger.info(f"Found camera: {info}")

        return available_cameras

    def start(self, source=0):
        """初始化摄像头"""
        try:
            # 先检查摄像头是否有效
            if not self._is_valid_camera(source):
                raise RuntimeError(f"Camera {source} is not available")

            # 如果已经初始化，先释放
            if self.cap is not None:
                self.release()
                time.sleep(0.5)

            self.logger.info(f"Initializing camera {source}")
            self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {source}")

            # 获取摄像头参数
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            # 设置缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

            # 读取测试帧
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read frame from camera")

            self.is_initialized = True
            self.logger.info(f"Camera initialized successfully: {self.width}x{self.height} @ {self.fps}fps")

        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            self.is_initialized = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            raise

    def read(self):
        """读取一帧"""
        if not self.is_initialized or self.cap is None:
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None

            return True, frame

        except Exception as e:
            self.logger.error(f"Error reading frame: {str(e)}")
            return False, None

    def get_info(self):
        """获取摄像头详细信息"""
        if not self.is_initialized or self.cap is None:
            return {
                'initialized': False,
                'width': 0,
                'height': 0,
                'fps': 0
            }

        return {
            'initialized': self.is_initialized,
            'width': self.width,
            'height': self.height,
            'fps': self.fps
        }

    def release(self):
        """释放摄像头资源"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_initialized = False
            self.logger.info("Camera released")
        except Exception as e:
            self.logger.error(f"Error releasing camera: {str(e)}")


class StreamCamera:
    url_cache_duration = InitialConfig.STREAM_URL_CACHE_DURATION
    _device_names = {}  # 类变量，用于存储设备名称映射
    _stream_url_cache = None  # 缓存的流地址
    _stream_url_expire = 0  # 流地址过期时间
    _url_lock = threading.Lock()  # URL缓存的线程锁

    def __init__(self):
        self.logger = logging.getLogger('StreamCamera')
        self._lock = threading.Lock()  # 添加线程锁
        self.is_running = False
        self.video_container = None
        self.audio_container = None
        self.audio_stream = None
        self.video_stream = None
        self.audio_thread = None
        self.is_initialized = False
        self.width = 1920
        self.height = 1080
        self.fps = 30
        self._audio_callback = None
        self._silence_threshold = 0.02
        self.analysis_enabled = False
        self.reconnect_attempts = 0
        self.max_reconnects = 3
        self.reconnect_delay = 2
        self.last_reconnect_time = 0

    def set_analysis_enabled(self, enabled: bool):
        """设置是否启用分析"""
        self.analysis_enabled = enabled

    def set_audio_callback(self, callback):
        """设置音频回调函数"""
        self._audio_callback = callback

    def get_info(self):
        """获取摄像头信息"""
        return {
            'initialized': self.is_initialized,
            'width': self.width,
            'height': self.height,
            'fps': self.fps
        }

    def audio_processing_loop(self):
        """改进的音频处理循环"""
        try:
            for frame in self.process_stream(self.audio_container, self.audio_stream):
                if not self.is_running:
                    break
                
                try:
                    # 转换音频帧为numpy数组
                    audio_data = frame.to_ndarray()
                    
                    # 确保数据是float32类型
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                        
                    # 规范化音频数据
                    if audio_data.max() > 1.0:
                        audio_data = audio_data / 32768.0
                        
                    # 确保数据形状正确
                    if audio_data.ndim == 1:
                        audio_data = audio_data.reshape(-1, 1)
                    
                    # 回调处理
                    if self._audio_callback and self.analysis_enabled:
                        self._audio_callback(audio_data)
                        
                except Exception as e:
                    self.logger.error(f"Error processing audio frame: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Audio processing loop error: {e}")

    @staticmethod
    def is_valid_camera(index):
        """公共方法：检查摄像头是否有效"""
        return True  # Stream cameras are always valid

    @classmethod
    def get_device_name(cls, index):
        """公共方法：获取设备名称"""
        return cls._device_names.get(str(index), f'Stream Camera {index}')

    @classmethod
    def update_device_names(cls, device_names):
        """更新设备名称映射"""
        cls._device_names = device_names

    @staticmethod
    def list_cameras():
        """列出所有可用摄像头"""
        # 每次获取设备列表时都重新获取流地址以验证有效性
        try:
            url = StreamCamera.get_stream_url()
            if url:
                return [{
                    'index': 0,
                    'name': StreamCamera._device_names.get('0', 'Stream Camera 0'),
                    'width': 1920,
                    'height': 1080,
                    'fps': 30
                }]
        except Exception as e:
            logging.error(f"Error getting stream URL: {e}")
        return []

    @classmethod
    def get_stream_url(cls) -> Optional[str]:
        """获取流媒体地址,带缓存机制"""
        current_time = time.time()

        with cls._url_lock:
            # 检查是否有有效的缓存URL
            if (cls._stream_url_cache and
                    current_time < cls._stream_url_expire):
                return cls._stream_url_cache

            # 检查请求频率限制
            if (current_time - cls._stream_url_expire) < 2:
                if cls._stream_url_cache:  # 如果有旧缓存,先返回旧的
                    return cls._stream_url_cache
                time.sleep(2)  # 强制等待最小间隔

            try:
                url = "https://open.ys7.com/api/lapp/v2/live/address/get"
                params = {
                    "accessToken": InitialConfig.STREAM_CAMERA_ACCESS_TOKEN,
                    "deviceSerial": InitialConfig.STREAM_CAMERA_SERIAL,
                    "protocol": InitialConfig.STREAM_CAMERA_PROTOCOL,
                    "supportH265": 0
                }

                response = requests.post(url, params=params, verify=False)
                response.raise_for_status()
                data = response.json()

                # 解析返回数据中的过期时间
                expireTime = data.get('data', {}).get('expireTime')
                if expireTime:
                    try:
                        # 将过期时间转换为时间戳
                        expire_timestamp = time.mktime(time.strptime(expireTime, "%Y-%m-%d %H:%M:%S"))
                    except:
                        # 如果转换失败，使用默认缓存时间
                        expire_timestamp = current_time + cls.url_cache_duration
                else:
                    expire_timestamp = current_time + cls.url_cache_duration

                stream_url = data.get('data', {}).get('url')
                if stream_url:
                    cls._stream_url_cache = stream_url
                    cls._stream_url_expire = expire_timestamp
                    return stream_url

            except Exception as e:
                logging.error(f"Failed to get stream URL: {e}")
                if cls._stream_url_cache:  # 如果请求失败但有缓存,返回缓存的URL
                    return cls._stream_url_cache
                raise

            return None

    def process_stream(self, container, stream):
        """单独处理视频或音频流"""
        try:
            for packet in container.demux(stream):
                if not self.is_running:
                    break
                for frame in packet.decode():
                    yield frame
        except Exception as e:
            print(f"Stream processing error: {e}")

    def start(self, source=0):
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                with self._lock:
                    # 获取流媒体地址
                    rtmp_url = self.get_stream_url()

                    # 初始化视频容器(缺少这部分)
                    self.video_container = av.open(rtmp_url, options={
                        'rtsp_transport': 'tcp',
                        'stimeout': '5000000',
                        'reconnect': '1',
                        'reconnect_streamed': '1',
                        'reconnect_delay_max': '2'
                    })
                    
                    # 初始化视频流(缺少这部分)
                    self.video_stream = self.video_container.streams.video[0]
                    self.is_running = True
                    self.is_initialized = True  # 设置初始化标志

                    # 创建独立的音频容器
                    self.audio_container = av.open(rtmp_url, options={
                        'rtsp_transport': 'tcp',
                        'stimeout': '5000000',
                        'reconnect': '1',
                        'reconnect_streamed': '1',
                        'reconnect_delay_max': '2'
                    })

                    # 初始化音频流
                    self.audio_stream = self.audio_container.streams.audio[0]
                    self.logger.info(f"Audio stream initialized: {self.audio_stream.rate}Hz, "
                                f"{self.audio_stream.channels} channels")

                    # 启动音频处理线程
                    self.audio_thread = threading.Thread(target=self.audio_processing_loop)
                    self.audio_thread.daemon = True
                    self.audio_thread.start()

                return
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Stream initialization error (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(1)  # 重试延迟
                
        raise RuntimeError("Failed to initialize stream after maximum retries")

    def read(self):
        """读取视频帧，包含自动重连逻辑"""
        if not self.is_running:
            self.reconnect()  # 如果未运行则尝试重连
            return False, None

        try:
            if self.video_stream is None or self.video_container is None:
                if time.time() - self.last_reconnect_time > self.reconnect_delay:
                    self.reconnect()
                return False, None

            for packet in self.video_container.demux(self.video_stream):
                for frame in packet.decode():
                    if isinstance(frame, av.VideoFrame):
                        return True, frame.to_ndarray(format='bgr24')
                break  # 只读取一帧就退出

            return False, None

        except (av.error.OSError, av.error.InvalidDataError) as e:
            self.logger.error(f"Stream read error: {e}")
            self.video_stream = None  # 触发重连
            return False, None

    def reconnect(self):
        """改进的重连机制"""
        try:
            current_time = time.time()
            if (current_time - self.last_reconnect_time) < self.reconnect_delay:
                return False
                
            self.last_reconnect_time = current_time
            self.reconnect_attempts += 1
            
            if self.reconnect_attempts > self.max_reconnects:
                self.logger.error("Max reconnection attempts reached")
                return False
                
            self.cleanup()
            
            rtmp_url = self.get_stream_url()
            if not rtmp_url:
                return False
                
            # 重新初始化视频和音频容器
            self.video_container = av.open(rtmp_url, options={
                'rtsp_transport': 'tcp',
                'stimeout': '5000000',
                'reconnect_streamed': '1',
                'reconnect_delay_max': '2'
            })
            
            self.audio_container = av.open(rtmp_url, options={
                'rtsp_transport': 'tcp',
                'stimeout': '5000000',
                'reconnect_streamed': '1',
                'reconnect_delay_max': '2'
            })
            
            # 重新获取视频和音频流
            self.video_stream = self.video_container.streams.video[0]
            try:
                self.audio_stream = self.audio_container.streams.audio[0]
                self.audio_thread = threading.Thread(target=self.audio_processing_loop)
                self.audio_thread.daemon = True
                self.audio_thread.start()
            except Exception as e:
                self.logger.error(f"Audio reconnection failed: {e}")
                
            self.is_running = True
            self.logger.info("Stream reconnected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            return False

    def cleanup(self) -> None:
        """改进的资源清理"""
        self.is_running = False
        self.is_initialized = False  # 重置初始化状态

        # 停止音频线程
        if hasattr(self, 'audio_thread') and self.audio_thread:
            try:
                self.audio_thread.join(timeout=1.0)
            except Exception as e:
                self.logger.error(f"Error stopping audio thread: {e}")

        # 关闭容器
        for container_name in ['video_container', 'audio_container']:
            if hasattr(self, container_name):
                try:
                    container = getattr(self, container_name)
                    if container:
                        container.close()
                except Exception as e:
                    self.logger.error(f"Error closing {container_name}: {e}")
                finally:
                    setattr(self, container_name, None)

        # 重置其他状态
        self.video_stream = None
        self.audio_stream = None
        self.reconnect_attempts = 0
        cv2.destroyAllWindows()
