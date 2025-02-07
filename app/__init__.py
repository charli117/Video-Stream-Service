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
        mic = Microphone()
        mic.set_analysis_enabled(True)  # 确保开启音频提取
        available_devices = mic.list_devices()
        
        if available_devices:
            # 确保设备列表不为空
            device_index = available_devices[0]['index']
            app.logger.info(f"Found {len(available_devices)} audio devices")
            app.logger.info(f"Using audio device: {available_devices[0]['name']} (index: {device_index})")
            
            # 初始化 Microphone 并开启音频分析功能
            mic.start(device_index=device_index)  # 请根据实际设备调整索引

            audio_analyzer.microphone = mic
            audio_analyzer.start(device_index=device_index)
            app.logger.info("Audio analyzer started successfully.")
        else:
            app.logger.warning("No audio input devices found, audio analysis will be disabled")
            
    except Exception as e:
        app.logger.error(f"Failed to initialize analyzers: {str(e)}")
        app.logger.error("Failed to start audio analyzer.")  # Pc548
        # 确保即使音频初始化失败，视频功能仍然可用
        if not video_analyzer.is_running:
            app.logger.error("Video analyzer failed to start")
        
    return app
