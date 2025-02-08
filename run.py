import os
import logging
from app import create_app
from app.socket import socketio

if __name__ == '__main__':
    # 确保必要的目录存在
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static/output', exist_ok=True)

    app = create_app()
    
    # 降低日志级别
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # 增加线程数
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=10)
    
    # 优化 socketio 配置
    socketio.run(app, 
        host='0.0.0.0',
        port=5008,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True)
