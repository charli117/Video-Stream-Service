import os


class InitialConfig:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev'

    # 视频分析配置
    MAX_QUEUE_SIZE = 32
    SCALE_PERCENT = 80
    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_FPS = 30
    MAX_DISPLAY_HEIGHT = 720
    MAX_SAVED_IMAGES = 1000

    # Audio Configuration
    AUDIO_SAMPLE_RATE = 44100
    AUDIO_CHUNK_SIZE = 4096
    AUDIO_CHANNELS = 1
    AUDIO_CHANGE_THRESHOLD = 0.5  # 降低阈值使检测更敏感
    MAX_SAVED_FILES = 1000
    
    # 日志配置
    LOG_PATH = 'logs/video_analyzer.log'
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 静态文件配置
    OUTPUT_DIR = './static/output'
