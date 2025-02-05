import os
import logging
from flask import Flask
from config import InitialConfig
from app.analyzer import VideoAnalyzer, AudioAnalyzer

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
        video_analyzer.start(InitialConfig.DEFAULT_CAMERA_INDEX)
        audio_analyzer.start()
    except Exception as e:
        app.logger.error(f"Failed to initialize analyzers: {str(e)}")

    return app
