import time
import logging
import traceback
import pyaudio
import sounddevice as sd
from config import InitialConfig
import numpy as np
from queue import Queue, Empty
import threading
import av


class Microphone:
    _device_names = {}  # 类变量，用于存储设备名称映射
    logger = logging.getLogger('Microphone')  # 定义类级别的 logger

    def __init__(self):
        self.stream = None
        self.audio = None
        self.current_device = 0
        self.sample_rate = InitialConfig.AUDIO_SAMPLE_RATE
        self.channels = InitialConfig.AUDIO_CHANNELS
        self.chunk_size = InitialConfig.AUDIO_CHUNK_SIZE
        self.is_initialized = False
        self.analysis_enabled = False
        self.audio_queue = Queue(maxsize=100)
        # 添加流式处理相关属性
        self.is_stream_mode = InitialConfig.CAMERA_TYPE == 'stream'
        self.stream_audio_container = None
        self.stream_audio_thread = None

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
            Microphone.logger.error(f"Error listing audio devices: {str(e)}")  # 使用类级别的 logger
            return []

    def start(self, device_index=None):
        """启动音频设备"""
        try:
            self.logger.info("Initializing audio device")
            if self.is_initialized:
                self.release()  # 确保先释放现有资源
                time.sleep(0.5)  # 等待资源释放
            
            if self.is_stream_mode:
                # 使用StreamCamera的音频流
                from .camera import StreamCamera
                stream_camera = StreamCamera()
                stream_url = stream_camera.get_stream_url()
                if not stream_url:
                    raise ValueError("Failed to get stream URL")
                
                # 初始化音频容器
                self.stream_audio_container = av.open(stream_url, options={
                    'rtsp_transport': 'tcp',
                    'stimeout': '5000000'
                })
                
                # 获取音频流
                audio_stream = self.stream_audio_container.streams.audio[0]
                self.sample_rate = audio_stream.rate
                self.channels = audio_stream.channels
                
                # 启动音频处理线程
                self.stream_audio_thread = threading.Thread(target=self._stream_audio_processing)
                self.stream_audio_thread.daemon = True
                self.stream_audio_thread.start()
                
            else:
                # 使用传入的device_index或默认值    
                device_index = device_index if device_index is not None else self.current_device
                self.current_device = device_index

                if not Microphone.is_valid_device(device_index):
                    raise ValueError(f"Invalid audio device index: {device_index}")

                def audio_callback(in_data, frame_count, time_info, status):
                    try:
                        # 将音频数据放入队列
                        if self.analysis_enabled and not self.audio_queue.full():
                            audio_data = np.frombuffer(in_data, dtype=np.float32)
                            self.audio_queue.put_nowait(audio_data)
                    except:
                        pass
                    return (in_data, pyaudio.paContinue)

                self.audio = pyaudio.PyAudio()
                self.stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    input_device_index=device_index,
                    stream_callback=audio_callback
                )
                
                # 确保流被正确打开
                if not self.stream.is_active():
                    self.stream.start_stream()
                
            self.is_initialized = True
            self.logger.info(f"Audio device initialized successfully: {self.channels} channels @ {self.sample_rate}Hz")

        except Exception as e:
            self.logger.error(f"Error initializing audio device: {str(e)}")
            self.release()
            raise

    def _stream_audio_processing(self):
        """流式音频处理线程"""
        try:
            audio_stream = self.stream_audio_container.streams.audio[0]
            
            for frame in self.stream_audio_container.decode(audio_stream):
                if not self.is_initialized:
                    break
                    
                try:
                    # 转换音频帧为numpy数组
                    audio_data = frame.to_ndarray()
                    
                    # 确保数据类型为float32
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    
                    # 归一化
                    if audio_data.max() > 1.0:
                        audio_data = audio_data / 32768.0
                    
                    # 放入队列
                    if self.analysis_enabled and not self.audio_queue.full():
                        self.audio_queue.put_nowait(audio_data)
                        
                except Exception as e:
                    self.logger.error(f"Error processing audio frame: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Stream audio processing error: {str(e)}")
            
    def read(self, frames=None, chunk_size=None):
        """从队列中读取音频数据"""
        if not self.analysis_enabled:
            return False, None

        if not self.is_initialized or self.stream is None:
            self.logger.error("Microphone未初始化")
            return False, None

        try:
            # 检查stream是否active
            if not self.stream.is_active():
                self.logger.warning("Audio stream is not active, attempting to restart")
                self.start(self.current_device)  # 尝试重新启动
                if not self.is_initialized or self.stream is None:
                    self.logger.error("Failed to restart audio stream")
                    return False, None

            read_frames = frames if frames else chunk_size if chunk_size else self.chunk_size
            data, overflowed = self.stream.read(read_frames)
            if overflowed:
                self.logger.warning("Audio input buffer overflowed")

            # 将返回数据转换为 numpy.ndarray
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            # 如果数据类型不是float32，转换为float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # 归一化音频数据
            if data.max() > 1.0:
                data = data / 32768.0

            return True, data

        except Empty:
            return False, None
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
                # 原有的释放逻辑
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                if self.audio:
                    self.audio.terminate()
                    self.audio = None
                
            self.is_initialized = False
            self.logger.info("Audio device released")
            
        except Exception as e:
            self.logger.error(f"Error releasing audio device: {str(e)}")
