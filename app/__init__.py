import os
import logging
from flask import Flask
from config import InitialConfig
from app.analyzer import VideoAnalyzer, AudioAnalyzer
from app.camera import Camera
from app.microphone import Microphone

# 配置日志
logging.basicConfig(level=InitialConfig.LOG_LEVEL, format=InitialConfig.LOG_FORMAT)

# 创建全局分析器实例
video_analyzer = VideoAnalyzer()
audio_analyzer = AudioAnalyzer()


def create_app():
    app = Flask(__name__,
                template_folder=os.path.abspath('templates'),
                static_folder=os.path.abspath('static'))

    # 注册蓝图
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # 初始化分析器
    try:
        # 启动视频分析器
        video_analyzer.start(InitialConfig.DEFAULT_CAMERA_INDEX)
        
        # 初始化音频设备
        if 'audio' in InitialConfig.ANALYZER_TYPE:
            mic = Microphone()
            # 如果是流媒体模式，不使用本地设备索引
            if InitialConfig.CAMERA_TYPE == 'stream':
                mic.start()
                audio_analyzer.microphone = mic
                audio_analyzer.start()
                app.logger.info("Audio analyzer initialized in stream mode")
            else:
                available_devices = mic.list_devices()
                if available_devices:
                    device_index = available_devices[0]['index']
                    app.logger.info(f"Using audio device: {available_devices[0]['name']} (index: {device_index})")
                    audio_analyzer.current_device = device_index
                    audio_analyzer.microphone = mic
                    audio_analyzer.start(device_index)
                    app.logger.info("Audio analyzer initialized successfully")
                else:
                    app.logger.warning("No audio input devices found")
    except Exception as e:
        app.logger.error(f"Failed to initialize audio analyzer: {str(e)}")
        if not video_analyzer.is_running:
            app.logger.error("Video analyzer failed to start")
        
    return app
