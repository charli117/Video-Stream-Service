import os
from app import create_app
from config import InitialConfig

if __name__ == '__main__':
    # 确保必要的目录存在
    os.makedirs('logs', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/output', exist_ok=True)

    app = create_app()
    # 始终启动 Web 服务
    app.run(host='0.0.0.0', port=5008, debug=False)
