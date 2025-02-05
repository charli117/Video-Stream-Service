import cv2
import time
import logging
from config import InitialConfig


class Camera:
    _device_names = {}  # 类变量，用于存储设备名称映射

    def __init__(self):
        self.cap = None
        self.logger = logging.getLogger('Camera')
        self.width = None
        self.height = None
        self.fps = InitialConfig.DEFAULT_FPS
        self.is_initialized = False
        self._buffer_size = InitialConfig.MAX_QUEUE_SIZE

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
            if Camera._is_valid_camera(i):
                cap = cv2.VideoCapture(i)
                info = {
                    'index': i,
                    'name': Camera._device_names.get(str(i), f'Camera {i}'),
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
