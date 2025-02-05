import logging
import os
from flask import Blueprint, render_template, Response, jsonify, request
from app import video_analyzer, audio_analyzer
from app.camera import Camera
from app.analyzer import AudioAnalyzer

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
        return Response(video_analyzer.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video feed: {str(e)}")
        return "Video feed error", 500


@main_bp.route('/api/cameras')
def list_cameras():
    """获取可用摄像头列表"""
    try:
        cameras = Camera.list_cameras()
        current_camera = video_analyzer.video_source

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

        if not Camera._is_valid_camera(camera_index):
            return jsonify({
                'success': False,
                'error': f'Camera {camera_index} is not available'
            }), 400

        success = video_analyzer.switch_camera(camera_index)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@main_bp.route('/api/devices')
def list_devices():
    """获取可用设备列表"""
    try:
        cameras = Camera.list_cameras()
        audio_devices = AudioAnalyzer.list_devices()
        
        return jsonify({
            'cameras': cameras,
            'audioDevices': audio_devices,
            'currentCamera': video_analyzer.video_source,
            'currentAudioDevice': audio_analyzer.current_device
        })
    except Exception as e:
        logger.error(f"Error listing devices: {str(e)}")
        return jsonify({
            'cameras': [],
            'audioDevices': [],
            'currentCamera': None,
            'currentAudioDevice': None,
            'error': str(e)
        }), 500

@main_bp.route('/api/devices/switch', methods=['POST'])
def switch_devices():
    """切换设备"""
    try:
        data = request.get_json()
        camera_index = int(data.get('camera_index', 0))
        audio_index = int(data.get('audio_index', 0))

        if not Camera._is_valid_camera(camera_index):
            return jsonify({
                'success': False,
                'error': f'Camera {camera_index} is not available'
            }), 400

        camera_success = video_analyzer.switch_camera(camera_index)
        audio_success = audio_analyzer.switch_device(audio_index)

        if not camera_success or not audio_success:
            return jsonify({
                'success': False,
                'error': 'Failed to switch one or more devices'
            }), 400

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@main_bp.route('/api/device_names', methods=['POST'])
def update_device_names():
    """更新设备名称"""
    try:
        data = request.get_json()
        device_names = data.get('deviceNames', {})
        
        Camera.update_device_names(device_names.get('video', {}))
        AudioAnalyzer.update_device_names(device_names.get('audio', {}))
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@main_bp.route('/api/status')
def get_status():
    try:
        status = {
            'is_running': video_analyzer.is_running,
            'current_camera': video_analyzer.video_source,
            'fps': video_analyzer.fps,
            'camera_info': {},
            'analysis_enabled': video_analyzer.analysis_enabled,
            'frame_changed': not video_analyzer.change_queue.empty(),
            'frame_changes': [],
            'audio_enabled': audio_analyzer.is_running,
            'audio_changes': []
        }

        if video_analyzer.camera and video_analyzer.camera.is_initialized:
            status['camera_info'] = {
                'width': video_analyzer.camera.width,
                'height': video_analyzer.camera.height,
                'fps': video_analyzer.camera.fps,
                'initialized': True
            }

        if status['analysis_enabled']:
            try:
                output_dir = video_analyzer.output_dir
                image_files = sorted(
                    [f for f in os.listdir(output_dir) if f.endswith('.jpg')],
                    key=lambda x: os.path.getmtime(os.path.join(output_dir, x)),
                    reverse=True
                )[:10]

                for img_file in image_files:
                    file_path = os.path.join(output_dir, img_file)
                    timestamp = os.path.getmtime(file_path)
                    status['frame_changes'].append({
                        'time': timestamp,
                        'image_url': f'/static/output/{img_file}'
                    })
            except Exception as e:
                logger.error(f"Error getting frame changes: {str(e)}")

        # 添加音频变化记录
        if audio_analyzer.is_running:
            try:
                output_dir = audio_analyzer.output_dir
                audio_files = sorted(
                    [f for f in os.listdir(output_dir) if f.endswith('.wav')],
                    key=lambda x: os.path.getmtime(os.path.join(output_dir, x)),
                    reverse=True
                )[:10]

                for audio_file in audio_files:
                    file_path = os.path.join(output_dir, audio_file)
                    timestamp = os.path.getmtime(file_path)
                    status['audio_changes'].append({
                        'time': timestamp,
                        'audio_url': f'/static/output/{audio_file}'
                    })
            except Exception as e:
                logger.error(f"Error getting audio changes: {str(e)}")

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


@main_bp.route('/api/toggle_analysis', methods=['POST'])
def toggle_analysis():
    data = request.get_json()
    enabled = data.get('enabled', False)
    success = video_analyzer.toggle_analysis(enabled)
    return jsonify({'success': success, 'analysis_enabled': video_analyzer.analysis_enabled})


@main_bp.route('/api/toggle_audio', methods=['POST'])
def toggle_audio():
    try:
        data = request.get_json()
        enabled = data.get('enabled', False)

        if enabled:
            audio_analyzer.start()
        else:
            audio_analyzer.stop()

        return jsonify({
            'success': True,
            'audio_enabled': enabled
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@main_bp.route('/api/camera_names', methods=['POST'])
def update_camera_names():
    """更新摄像头名称"""
    try:
        data = request.get_json()
        device_names = data.get('deviceNames', {})
        # 更新全局摄像头名称映射
        Camera.update_device_names(device_names)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
