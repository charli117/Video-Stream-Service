import os
import logging

from flask import Flask
from app.analyzer import VideoAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 创建全局analyzer实例
analyzer = VideoAnalyzer()


def create_app():
    app = Flask(__name__,
                template_folder=os.path.abspath('templates'),
                static_folder=os.path.abspath('static'))

    # 注册蓝图
    from app.routes import main_bp

    app.register_blueprint(main_bp)

    # 只在启动时初始化默认摄像头
    try:
        analyzer.start(0)  # 默认使用0号摄像头
    except Exception as e:
        app.logger.error(f"Failed to initialize default camera: {str(e)}")

    return app
