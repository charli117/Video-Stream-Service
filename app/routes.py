from flask import Blueprint, render_template, Response, jsonify, request
from app import analyzer
from app.camera import Camera
import logging

# 创建蓝图
main_bp = Blueprint('main', __name__)
logger = logging.getLogger('Camera')


@main_bp.route('/')
def index():
    """主页"""
    return render_template('index.html')


@main_bp.route('/video_feed')
def video_feed():
    """视频流"""
    try:
        return Response(analyzer.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video feed: {str(e)}")
        return "Video feed error", 500


@main_bp.route('/api/cameras')
def list_cameras():
    """获取可用摄像头列表"""
    try:
        cameras = Camera.list_cameras()

        # 添加当前使用的摄像头信息
        current_camera = analyzer.video_source

        # 确保返回正确的数据结构
        return jsonify({
            'cameras': cameras,
            'current': current_camera
        })
    except Exception as e:
        logger.error(f"Error listing cameras: {str(e)}")
        return jsonify({
            'cameras': [],
            'current': None,
            'error': str(e)
        }), 500


@main_bp.route('/api/camera/switch', methods=['POST'])
def switch_camera():
    """切换摄像头"""
    try:
        data = request.get_json()
        camera_index = int(data.get('camera_index', 0))

        # 先检查摄像头是否可用
        if not Camera._is_valid_camera(camera_index):
            return jsonify({
                'success': False,
                'error': f'Camera {camera_index} is not available'
            }), 400

        success = analyzer.switch_camera(camera_index)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@main_bp.route('/api/status')
def get_status():
    """获取当前状态"""
    try:
        status = {
            'is_running': analyzer.is_running,
            'current_camera': analyzer.video_source,
            'fps': analyzer.fps,
            'camera_info': {}
        }

        # 获取摄像头信息
        if analyzer.camera and analyzer.camera.is_initialized:
            status['camera_info'] = {
                'width': analyzer.camera.width,
                'height': analyzer.camera.height,
                'fps': analyzer.camera.fps,
                'initialized': True
            }

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({
            'error': str(e),
            'is_running': False,
            'camera_info': {
                'initialized': False
            }
        }), 500
