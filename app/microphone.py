import time
import logging
import pyaudio
import sounddevice as sd
from config import InitialConfig
import numpy as np
from queue import Queue, Empty
import threading
import av
from .camera import StreamCamera


class Microphone:
    _device_names = {}  # 类变量，用于存储设备名称映射
    logger = logging.getLogger('Microphone')  # 定义类级别的 logger

    def __init__(self):
        self.stream = None
        self.audio = None
        self.current_device = 0
        self.sample_rate = InitialConfig.AUDIO_SAMPLE_RATE
        self._channels = None
        self.chunk_size = InitialConfig.AUDIO_CHUNK_SIZE
        self.is_initialized = False
        self.analysis_enabled = False
        self.audio_queue = Queue(maxsize=50)
        self.is_stream_mode = InitialConfig.CAMERA_TYPE == 'stream'
        self.stream_audio_container = None
        self.stream_audio_thread = None

    @property
    def channels(self):
        if self._channels is None:
            try:
                devices = sd.query_devices()
                device_info = devices[self.current_device]
                self._channels = device_info['max_input_channels']
            except Exception as e:
                self.logger.warning(f"Failed to get device channels, using default value 1: {e}")
                self._channels = 1
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    def set_analysis_enabled(self, enabled: bool):
        """设置是否启用音频提取与分析"""
        self.analysis_enabled = enabled

    @staticmethod
    def is_valid_device(index):
        """公共方法：检查音频设备是否有效"""
        return Microphone._is_valid_device(index)

    @classmethod
    def get_device_name(cls, index):
        """公共方法：获取设备名称"""
        return cls._device_names.get(str(index), f'Audio Device {index}')

    @classmethod
    def update_device_names(cls, device_names):
        """更新设备名称映射"""
        cls._device_names = device_names

    @classmethod
    def list_devices(cls):
        """列出所有可用音频输入设备"""
        try:
            devices = sd.query_devices()
            available_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    info = {
                        'index': i,
                        'name': Microphone._device_names.get(str(i), device['name']),
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    }
                    available_devices.append(info)
            return available_devices
        except Exception as e:
            Microphone.logger.error(f"Error listing audio devices: {str(e)}")
            return []

    def start(self, device_index=None):
        """启动音频设备"""
        try:
            self.logger.info("Initializing audio device")
            if self.is_initialized:
                self.release()  # 释放现有资源
                time.sleep(0.5)
            
            if self.is_stream_mode:
                # 流媒体模式下，忽略传入的 device_index，直接使用流媒体音频数据
                stream_camera = StreamCamera()
                stream_url = stream_camera.get_stream_url()
                if not stream_url:
                    raise ValueError("Failed to get stream URL")
                
                # 使用流媒体音频容器
                self.stream_audio_container = av.open(stream_url, options={
                    'rtsp_transport': 'tcp',
                    'stimeout': '5000000',
                    'reconnect_streamed': '1',
                    'reconnect_delay_max': '2'
                })
                audio_stream = self.stream_audio_container.streams.audio[0]
                self.sample_rate = audio_stream.rate
                self.channels = audio_stream.channels
                self.logger.info(f"Stream audio initialized: {self.sample_rate}Hz, {self.channels} channels")
                
                self.stream_audio_thread = threading.Thread(target=self._stream_audio_processing)
                self.stream_audio_thread.daemon = True
                self.stream_audio_thread.start()
                
            else:
                # 本地模式：使用传入的 device_index 或默认值
                device_index = device_index if device_index is not None else self.current_device
                self.current_device = device_index

                if not Microphone.is_valid_device(device_index):
                    raise ValueError(f"Invalid audio device index: {device_index}")

                devices = sd.query_devices()
                device_info = devices[device_index]
                self._channels = device_info['max_input_channels']
                self.logger.info(f"Detected audio device with {self._channels} channels")

                self.audio = pyaudio.PyAudio()
                self.stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    input_device_index=device_index,
                )
                if not self.stream.is_active():
                    self.stream.start_stream()
                
            self.is_initialized = True
            self.logger.info(f"Audio device initialized successfully: {self.channels} channels @ {self.sample_rate}Hz")

        except Exception as e:
            self.logger.error(f"Error initializing audio device: {str(e)}")
            self.release()
            raise

    # 在流模式下，确保读取数据后做归一化、调整形状
    def read(self, frames=None, chunk_size=None):
        """从队列或音频流中读取音频数据"""
        try:
            if not self.is_initialized:
                self.logger.info("Microphone未初始化，正在初始化...")
                self.start()  # 此处不传 device_index，保证流媒体模式正确
                time.sleep(0.1)
                if not self.is_initialized:
                    self.logger.error("初始化失败")
                    return False, None

            if self.is_stream_mode:
                try:
                    # 从队列中超时获取数据
                    data = self.audio_queue.get_nowait()
                except Empty:
                    return False, None
            else:
                if not self.stream or not self.stream.is_active():
                    self.logger.warning("音频流未激活，尝试重新初始化...")
                    self.release()
                    time.sleep(0.1)
                    self.start(self.current_device)
                    if not self.stream or not self.stream.is_active():
                        return False, None

                try:
                    read_frames = frames if frames else chunk_size if chunk_size else self.chunk_size
                    raw_data = self.stream.read(read_frames, exception_on_overflow=False)
                    data = np.frombuffer(raw_data, dtype=np.float32)
                except Exception as e:
                    self.logger.error(f"读取音频流错误: {str(e)}")
                    self.is_initialized = False
                    return False, None

            return True, data

        except Exception as e:
            self.logger.error(f"音频读取错误: {str(e)}")
            return False, None

    def get_info(self):
        """获取音频设备详细信息"""
        if not self.is_initialized or (not self.stream and not self.stream_audio_container):
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
        """释放资源"""
        try:
            if self.is_stream_mode:
                self.is_initialized = False
                if self.stream_audio_container:
                    self.stream_audio_container.close()
                    self.stream_audio_container = None
                if self.stream_audio_thread and self.stream_audio_thread.is_alive():
                    self.stream_audio_thread.join(timeout=1.0)
            else:
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                if self.audio:
                    self.audio.terminate()
                    self.audio = None
            self.is_initialized = False
        except Exception as e:
            self.logger.error(f"Error releasing audio device: {str(e)}")

    def _stream_audio_processing(self):
        """流式音频处理线程"""
        try:
            for frame in self._process_stream(self.stream_audio_container,
                                              self.stream_audio_container.streams.audio[0]):
                if not self.is_initialized:
                    break

                try:
                    if not self.audio_queue.full():
                        self.audio_queue.put_nowait(frame.to_ndarray())
                except Exception as e:
                    self.logger.error(f"Error processing audio frame: {e}")
        except Exception as e:
            self.logger.error(f"Stream audio processing error: {e}")

    @staticmethod
    def _process_stream(container, stream):
        """单独处理视频或音频流"""
        try:
            for packet in container.demux(stream):
                for frame in packet.decode():
                    yield frame
        except Exception as e:
            print(f"Stream processing error: {e}")

    @staticmethod
    def _is_valid_device(index):
        try:
            devices = sd.query_devices()
            if 0 <= index < len(devices):
                device = devices[index]
                return device['max_input_channels'] > 0
            return False
        except Exception:
            return False
