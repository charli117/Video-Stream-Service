import numpy as np
import requests
import cv2
import subprocess


def get_video_stream_url(accessToken, deviceSerial, protocol):
    url = f"https://open.ys7.com/api/lapp/v2/live/address/get?accessToken={accessToken}&deviceSerial={deviceSerial}&protocol={protocol}&supportH265=0"
    response = requests.post(url, verify=False)
    response_dict = response.json()
    if response.status_code == 200 and response_dict.get('data'):
        return response_dict['data']


def get_video_stream(rtmp_url):
    # 使用 FFmpeg 捕获视频流
    command = [
        'ffmpeg',
        '-i', rtmp_url,  # 输入流
        '-f', 'image2pipe',  # 输出格式为图像管道
        '-pix_fmt', 'bgr24',  # OpenCV 支持的像素格式
        '-vcodec', 'rawvideo',  # 原始视频编码
        '-'
    ]

    # 启动 FFmpeg 进程
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 读取并显示视频流
    while True:
        # 从 FFmpeg 输出中读取一帧
        raw_frame = process.stdout.read(1920 * 1080 * 3)  # 根据分辨率调整
        if not raw_frame:
            break

        # 将原始数据转换为 OpenCV 帧
        frame = np.frombuffer(raw_frame, dtype='uint8').reshape((1080, 1920, 3))

        # 显示帧
        cv2.imshow('RTMP Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    process.terminate()
    cv2.destroyAllWindows()


rtmp = get_video_stream_url(
    accessToken="at.2t8jgfnf3bqga14nam9fug2q1mq8a1du-86trtghttp-0q9fgkc-ztbmnjfzp",
    deviceSerial="G92727684",
    protocol=3
)
get_video_stream(rtmp_url=rtmp['url'])