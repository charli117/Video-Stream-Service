import os


class InitialConfig:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev'

    # 流媒体摄像头配置，支持 stream、local
    CAMERA_TYPE = 'stream'
    STREAM_CAMERA_ACCESS_TOKEN = "at.bd1d0zsrdwjf0md3bpq23jky4v51x9xe-8s3fg03m8t-1kmlia0-mw6h93hot"
    STREAM_CAMERA_SERIAL = "G92729163"
    STREAM_CAMERA_PROTOCOL = 3
    STREAM_URL_CACHE_DURATION = 60 * 60  # 默认缓存1小时
    STREAM_URL_MIN_INTERVAL = 10  # 最小请求间隔(秒)

    # 分析配置,支持 audio、video 两种选择
    ANALYZER_TYPE = ['audio', 'video']

    # 视频分析配置
    MAX_QUEUE_SIZE = 32
    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_FPS = 30
    MAX_DISPLAY_HEIGHT = 720
    MAX_SAVED_IMAGES = 1000
    CHANGE_FRAME_THRESHOLD = 12
    BLUR_THRESHOLD = 150.0
    SIMILARITY_THRESHOLD = 0.80

    # 音频分析配置
    AUDIO_SAMPLE_RATE = 16000  # 降低采样率
    AUDIO_CHUNK_SIZE = 2048   # 降低块大小
    AUDIO_CHANGE_THRESHOLD = 0.90  # 降低阈值使检测更敏感
    MAX_SAVED_FILES = 1000
    MIN_DURATION = 1  # 最小持续时间
    
    # 日志配置
    LOG_PATH = 'logs/video_analyzer.log'
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 静态文件配置
    OUTPUT_DIR = './static/output'
