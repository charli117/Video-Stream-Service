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
            mic.start()  # 只初始化设备，不启用分析
            available_devices = mic.list_devices()
            
            if available_devices:
                device_index = available_devices[0]['index']
                app.logger.info(f"Using audio device: {available_devices[0]['name']} (index: {device_index})")
                
                # 设置音频设备
                audio_analyzer.current_device = device_index
                audio_analyzer.microphone = mic
                
                # 启动分析器但不开启分析
                audio_analyzer.start(device_index)
                app.logger.info("Audio analyzer initialized successfully")
            else:
                app.logger.warning("No audio input devices found")
    except Exception as e:
        app.logger.error(f"Failed to initialize audio analyzer: {str(e)}")
        if not video_analyzer.is_running:
            app.logger.error("Video analyzer failed to start")
        
    return app
