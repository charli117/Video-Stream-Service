import logging
import sounddevice as sd
from config import InitialConfig
import time


class Microphone:
    _device_names = {}  # 类变量，用于存储设备名称映射

    def __init__(self):
        self.stream = None
        self.logger = logging.getLogger('Microphone')
        self.current_device = 0
        self.sample_rate = InitialConfig.AUDIO_SAMPLE_RATE
        self.channels = InitialConfig.AUDIO_CHANNELS
        self.chunk_size = InitialConfig.AUDIO_CHUNK_SIZE
        self.is_initialized = False

    @staticmethod
    def is_valid_device(index):
        """公共方法：检查音频设备是否有效"""
        return Microphone._is_valid_device(index)

    @classmethod
    def get_device_name(cls, index):
        """公共方法：获取设备名称"""
        return cls._device_names.get(str(index), f'Audio Device {index}')

    @staticmethod
    def _is_valid_device(index):
        """检查音频设备是否有效"""
        try:
            devices = sd.query_devices()
            if 0 <= index < len(devices):
                device = devices[index]
                return device['max_input_channels'] > 0
            return False
        except:
            return False

    @classmethod
    def update_device_names(cls, device_names):
        """更新设备名称映射"""
        cls._device_names = device_names

    @staticmethod
    def list_devices():
        """列出所有可用音频输入设备"""
        try:
            devices = sd.query_devices()
            available_devices = []
            logger = logging.getLogger('Microphone')

            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    info = {
                        'index': i,
                        'name': Microphone._device_names.get(str(i), device['name']),
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    }
                    available_devices.append(info)
                    logger.info(f"Found audio device: {info}")

            return available_devices
        except Exception as e:
            logging.error(f"Error listing audio devices: {str(e)}")
            return []

    def start(self, device_index=None):
        """初始化音频设备"""
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                if device_index is not None:
                    self.current_device = device_index

                # 先检查设备是否有效
                if not self._is_valid_device(self.current_device):
                    raise RuntimeError(f"Audio device {self.current_device} is not available")

                # 如果已经初始化，先释放
                if self.stream is not None:
                    self.release()

                # 获取设备信息
                device_info = sd.query_devices(self.current_device)
                self.channels = min(device_info['max_input_channels'], InitialConfig.AUDIO_CHANNELS)
                self.sample_rate = int(device_info['default_samplerate'])

                self.logger.info(f"Initializing audio device {self.current_device}, attempt {attempt + 1}")
                self.stream = sd.InputStream(
                    device=self.current_device,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size
                )
                self.stream.start()

                self.is_initialized = True
                self.logger.info(f"Audio device initialized successfully: {self.channels} channels @ {self.sample_rate}Hz")
                return  # Initialization successful, exit the retry loop

            except Exception as e:
                self.logger.error(f"Error initializing audio device (attempt {attempt + 1}/{max_retries}): {str(e)}")
                self.is_initialized = False
                if self.stream is not None:
                    self.stream.close()
                    self.stream = None
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)  # Wait before the next retry
                else:
                    raise  # Re-raise the exception if all retries failed

    def read(self, frames=None, chunk_size=None):
        """读取音频数据"""
        if not self.is_initialized or self.stream is None:
            return False, None

        try:
            # 如果没有指定 frames，则使用 chunk_size，如果 chunk_size 也没有指定，则使用 self.chunk_size
            read_frames = frames if frames else chunk_size if chunk_size else self.chunk_size
            data, overflowed = self.stream.read(read_frames)
            if overflowed:
                self.logger.warning("Audio input buffer overflowed")
            return True, data

        except Exception as e:
            self.logger.error(f"Error reading audio: {str(e)}")
            return False, None

    def get_info(self):
        """获取音频设备详细信息"""
        if not self.is_initialized or self.stream is None:
            return {
                'initialized': False,
                'channels': 0,
                'sample_rate': 0
            }

        return {
            'initialized': self.is_initialized,
            'channels': self.channels,
            'sample_rate': self.sample_rate
        }

    def release(self):
        """释放音频设备资源"""
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.is_initialized = False
            self.logger.info("Audio device released")
        except Exception as e:
            self.logger.error(f"Error releasing audio device: {str(e)}")
