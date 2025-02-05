import os


class Config:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev'

    # 视频分析配置
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    MAX_QUEUE_SIZE = 32
    CHANGE_THRESHOLD = 0.8

    # 日志配置
    LOG_PATH = 'logs/video_analyzer.log'
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# P070c
DEFAULT_CAMERA_INDEX = 0

# P4d37
MAX_SAVED_IMAGES = 1000

# P99fa
MAX_DISPLAY_HEIGHT = 720

# P5f61
SCALE_PERCENT = 80

# Pedcb
OUTPUT_DIR = './static/output'
