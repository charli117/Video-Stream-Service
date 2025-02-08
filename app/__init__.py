import os
import time
import logging
from flask import Flask
from config import InitialConfig
from app.analyzer import VideoAnalyzer, AudioAnalyzer
from app.camera import Camera
from app.microphone import Microphone
from app.socket import socketio  # 从新文件导入

# 配置日志
logging.basicConfig(level=InitialConfig.LOG_LEVEL, format=InitialConfig.LOG_FORMAT)

# 创建全局分析器实例
video_analyzer = VideoAnalyzer()
audio_analyzer = AudioAnalyzer()


def init_video_analyzer(app):
    """初始化视频分析器"""
    try:
        video_analyzer.start(InitialConfig.DEFAULT_CAMERA_INDEX)
        app.logger.info("Video analyzer started successfully")
    except Exception as e:
        app.logger.error(f"Video analyzer initialization failed: {str(e)}")


def init_audio_analyzer(app):
    """初始化音频分析器"""
    if 'audio' not in InitialConfig.ANALYZER_TYPE:
        return True
        
    retries = 3
    retry_delay = 3  # 增加重试延迟
    
    for i in range(retries):
        try:
            # 确保清理旧资源
            if audio_analyzer.microphone:
                audio_analyzer.microphone.release()
                time.sleep(1)
            
            mic = app.config.get('microphone')
            if not mic:
                raise RuntimeError("Microphone instance not found in app.config")
            
            # 列出可用设备并记录日志
            available_devices = mic.list_devices()
            app.logger.info(f"Available audio devices: {available_devices}")
            
            if InitialConfig.CAMERA_TYPE != 'stream':
                if not available_devices:
                    raise RuntimeError("No audio devices available")
                device_index = available_devices[0]['index']
                app.logger.info(f"Selected audio device index: {device_index}")
                mic.start(device_index)
            else:
                mic.start()
                
            time.sleep(1)
            
            if not mic.is_initialized:
                raise RuntimeError("Audio device initialization failed")
                
            audio_analyzer.microphone = mic
            audio_analyzer.start()
            app.logger.info("Audio analyzer started successfully")
            return True
            
        except Exception as e:
            app.logger.warning(f"Retry {i+1}/{retries} failed: {str(e)}")
            time.sleep(retry_delay)
            retry_delay *= 2  # 指数退避
    
    app.logger.error("All audio initialization attempts failed")
    return False


def create_app():
    app = Flask(__name__,
                template_folder=os.path.abspath('templates'),
                static_folder=os.path.abspath('static'))

    # 添加设备初始化状态追踪
    app.config['DEVICE_INIT_STATUS'] = {
        'camera': False,
        'audio': False,
        'error': None
    }

    try:
        # 初始化摄像头并保存到app.config
        if 'video' in InitialConfig.ANALYZER_TYPE:
            app.config['camera'] = Camera()  # 保存实例
            app.config['DEVICE_INIT_STATUS']['camera'] = True
            
        # 初始化音频并保存到app.config
        if 'audio' in InitialConfig.ANALYZER_TYPE:
            app.config['microphone'] = Microphone()  # 保存实例
            app.config['DEVICE_INIT_STATUS']['audio'] = True
            
    except Exception as e:
        app.config['DEVICE_INIT_STATUS']['error'] = str(e)
        app.logger.error(f"Device initialization failed: {e}")

    # 注册蓝图
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # 分别初始化视频与音频分析器,并传入相应的设备实例
    if 'video' in InitialConfig.ANALYZER_TYPE:
        video_analyzer.camera = app.config.get('camera')
        init_video_analyzer(app)
    
    if 'audio' in InitialConfig.ANALYZER_TYPE:
        audio_analyzer.microphone = app.config.get('microphone') 
        init_audio_analyzer(app)

    # 初始化 Socket.IO
    socketio.init_app(app, cors_allowed_origins="*")

    return app
