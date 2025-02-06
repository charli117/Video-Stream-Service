import cv2
import time
import logging
import numpy as np
import requests
import av
import sounddevice as sd
import threading
from typing import Optional
from config import InitialConfig


class Camera:
    def __init__(self):
        self.logger = logging.getLogger('Camera')


class LocalCamera:
    _device_names = {}  # 类变量，用于存储设备名称映射

    def __init__(self):
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
                    'fps': int(cap.get(cv2.CAP_PROP_FPS))
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
    def __init__(self):
        self.is_running = False
        self.video_container = None
        self.audio_container = None
        self.audio_stream = None
        self.video_stream = None
        self.audio_thread = None

    @staticmethod
    def get_stream_url() -> Optional[str]:
        url = f"https://open.ys7.com/api/lapp/v2/live/address/get"
        params = {
            "accessToken": InitialConfig.STREAM_CAMERA_ACCESS_TOKEN,
            "deviceSerial": InitialConfig.STREAM_CAMERA_SERIAL,
            "protocol": InitialConfig.STREAM_CAMERA_PROTOCOL,
            "supportH265": 0
        }

        try:
            response = requests.post(url, params=params, verify=False)
            response.raise_for_status()
            data = response.json()
            return data.get('data', {}).get('url')
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get stream URL: {e}")

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

    def start_stream(self, rtmp_url: str, max_retries: int = 3) -> None:
        if not rtmp_url:
            raise ValueError("RTMP URL is required")

        retry_count = 0
        while retry_count < max_retries:
            try:
                # 创建视频容器
                self.video_container = av.open(rtmp_url, options={
                    'rtsp_transport': 'tcp',
                    'stimeout': '5000000'
                })

                # 创建音频容器
                self.audio_container = av.open(rtmp_url, options={
                    'rtsp_transport': 'tcp',
                    'stimeout': '5000000'
                })

                self.is_running = True

                # 获取视频流
                self.video_stream = self.video_container.streams.video[0]

                # 初始化音频流
                try:
                    self.audio_stream = self.audio_container.streams.audio[0]
                    print(f"Audio stream info: {self.audio_stream.rate}Hz, "
                          f"{self.audio_stream.channels} channels")

                    # 启动音频处理线程
                    self.audio_thread = threading.Thread(target=self.audio_processing_loop)
                    self.audio_thread.daemon = True
                    self.audio_thread.start()

                except Exception as e:
                    print(f"Audio initialization error: {e}")

                # 处理视频流
                for frame in self.process_stream(self.video_container, self.video_stream):
                    if isinstance(frame, av.VideoFrame):
                        img = frame.to_ndarray(format='bgr24')
                        cv2.imshow('RTMP Stream', img)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.is_running = False
                            break

                break  # 如果成功运行，跳出重试循环

            except av.error.OSError as e:
                print(f"Stream error (attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                time.sleep(2)
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
            finally:
                self.cleanup()

    def audio_processing_loop(self):
        """音频处理循环"""
        try:
            # 创建与输入流相同通道数的输出流
            stream = sd.RawOutputStream(
                samplerate=self.audio_stream.rate,
                channels=self.audio_stream.channels,
                dtype=np.float32
            )

            with stream:
                print(f"Audio output started: {stream.samplerate}Hz, {stream.channels} channels")

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

                        # 写入音频数据
                        stream.write(audio_data)

                    except Exception as e:
                        print(f"Audio frame processing error: {e}")
                        continue

        except Exception as e:
            print(f"Audio processing error: {e}")

    def cleanup(self) -> None:
        self.is_running = False

        if hasattr(self, 'video_container'):
            try:
                self.video_container.close()
            except Exception as e:
                print(f"Error closing video container: {e}")

        if hasattr(self, 'audio_container'):
            try:
                self.audio_container.close()
            except Exception as e:
                print(f"Error closing audio container: {e}")

        cv2.destroyAllWindows()
