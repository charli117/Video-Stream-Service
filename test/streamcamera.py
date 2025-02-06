import numpy as np
import requests
import cv2
import av
import time
import sounddevice as sd
import threading
from typing import Optional


class StreamCamera:
    def __init__(self):
        self.is_running = False
        self.video_container = None
        self.audio_container = None
        self.audio_stream = None
        self.video_stream = None
        self.enable_audio = True
        self.audio_thread = None

    @staticmethod
    def get_stream_url(access_token: str, device_serial: str, protocol: int) -> Optional[str]:
        if not all([access_token, device_serial, protocol]):
            raise ValueError("Invalid parameters")

        url = f"https://open.ys7.com/api/lapp/v2/live/address/get"
        params = {
            "accessToken": access_token,
            "deviceSerial": device_serial,
            "protocol": protocol,
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

                if self.enable_audio:
                    # 创建音频容器
                    self.audio_container = av.open(rtmp_url, options={
                        'rtsp_transport': 'tcp',
                        'stimeout': '5000000'
                    })

                self.is_running = True

                # 获取视频流
                self.video_stream = self.video_container.streams.video[0]

                # 初始化音频流
                if self.enable_audio:
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
                        self.enable_audio = False

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


def main():
    handler = StreamCamera()
    try:
        rtmp_url = handler.get_stream_url(
            access_token="at.bd1d0zsrdwjf0md3bpq23jky4v51x9xe-8s3fg03m8t-1kmlia0-mw6h93hot",
            device_serial="G92729163",
            protocol=3
        )
        handler.enable_audio = True  # 启用音频
        handler.start_stream(rtmp_url)
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
