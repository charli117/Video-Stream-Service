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
        
        # 获取可用的音频设备
        available_devices = Microphone.list_devices()
        
        if available_devices:
            # 确保设备列表不为空
            device_index = available_devices[0]['index']
            app.logger.info(f"Found {len(available_devices)} audio devices")
            app.logger.info(f"Using audio device: {available_devices[0]['name']} (index: {device_index})")
            
            # 设置音频设备索引并启动音频分析器
            audio_analyzer.current_device = device_index  # 先设置设备索引
            app.logger.info("Starting audio analyzer...")  # P0a33
            audio_analyzer.start()  # 然后启动分析器
            app.logger.info("Audio analyzer started successfully.")  # Pb2de
        else:
            app.logger.warning("No audio input devices found, audio analysis will be disabled")
            
    except Exception as e:
        app.logger.error(f"Failed to initialize analyzers: {str(e)}")
        app.logger.error("Failed to start audio analyzer.")  # Pc548
        # 确保即使音频初始化失败，视频功能仍然可用
        if not video_analyzer.is_running:
            app.logger.error("Video analyzer failed to start")
        
    return app
