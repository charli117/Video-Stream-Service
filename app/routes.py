import logging
import os
from flask import Blueprint, render_template, Response, jsonify, request
from app import video_analyzer, audio_analyzer
from app.camera import Camera
from app.microphone import Microphone
from config import InitialConfig

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


@main_bp.route('/api/devices')
def get_devices():
    """获取设备列表"""
    try:
        cameras = Camera.list_cameras()
        
        # 只在本地模式下获取音频设备
        audio_devices = []
        if InitialConfig.CAMERA_TYPE == 'local':
            audio_devices = Microphone.list_devices()
        
        # 根据 CAMERA_TYPE 返回不同的控件配置
        if InitialConfig.CAMERA_TYPE == 'local':
            controls = ['cameraSelect', 'audioSelect', 'Refresh Devices']
        else:
            controls = ['Refresh Devices', 'Open Analysis']
            
        return jsonify({
            'cameras': cameras,
            'audioDevices': audio_devices,
            'currentCamera': video_analyzer.video_source,
            'currentAudioDevice': audio_analyzer.current_device if InitialConfig.CAMERA_TYPE == 'local' else None,
            'controls': controls
        })
    except Exception as e:
        logger.error(f"Error getting devices: {str(e)}")
        return jsonify({'error': str(e)}), 500


@main_bp.route('/api/devices/switch', methods=['POST'])
def switch_devices():
    """切换设备"""
    try:
        data = request.get_json()
        camera_index = int(data.get('camera_index', 0))
        audio_index = int(data.get('audio_index', 0))

        if InitialConfig.CAMERA_TYPE == 'stream':
            logger.info(f"Switching to stream camera {camera_index}")
            camera_success = video_analyzer.switch_camera(camera_index)
            audio_success = True  # Stream cameras do not switch audio devices
        else:
            if not Camera.is_valid_camera(camera_index):
                return jsonify({
                    'success': False,
                    'error': f'Camera {camera_index} is not available'
                }), 400

            logger.info(f"Switching to camera {camera_index} and audio device {audio_index}")
            camera_success = video_analyzer.switch_camera(camera_index)
            audio_success = audio_analyzer.switch_audio(audio_index)

        logger.info(f"Camera switch success: {camera_success}, Audio switch success: {audio_success}")

        if not camera_success or not audio_success:
            return jsonify({
                'success': False,
                'error': 'Failed to switch one or more devices'
            }), 400

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error switching devices: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400


@main_bp.route('/api/device_names', methods=['POST'])
def update_device_names():
    """更新设备名称"""
    try:
        data = request.get_json()
        device_names = data.get('deviceNames', {})
        audio_device_names = data.get('audioDeviceNames', {})
        
        # 更新摄像头和音频设备名称映射
        Camera.update_device_names(device_names)
        Microphone.update_device_names(audio_device_names)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@main_bp.route('/api/status')
def get_status():
    """获取当前状态"""
    try:
        current_camera_index = str(video_analyzer.video_source)
        current_audio_index = str(audio_analyzer.current_device)

        # 初始化基本状态信息
        status = {
            'is_running': video_analyzer.is_running,
            'current_camera': video_analyzer.video_source,
            'current_camera_name': Camera.get_device_name(current_camera_index),
            'current_audio_device': audio_analyzer.current_device,
            'current_audio_name': Microphone.get_device_name(current_audio_index),
            'fps': video_analyzer.fps,
            'camera_info': {
                'initialized': False,
                'width': 0,
                'height': 0,
                'fps': 0,
                'name': Camera.get_device_name(current_camera_index)
            },
            'analysis_enabled': video_analyzer.analysis_enabled,
            'frame_changed': not video_analyzer.change_queue.empty(),
            'frame_changes': [],
            'audio_changes': []
        }

        # 更新摄像头信息
        if video_analyzer.camera and hasattr(video_analyzer.camera, 'camera'):
            camera_info = video_analyzer.camera.get_info()
            status['camera_info'].update(camera_info)

        # 获取音、视频帧变化记录
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
                logger.error(f"Error getting frame changes: {str(e)}")

        return jsonify(status)

    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({
            'error': str(e),
            'is_running': False,
            'camera_info': {
                'initialized': False,
                'name': "Not Connected"
            },
            'current_devices': {
                'camera': "Not Connected",
                'audio': "Not Connected"
            },
            'frame_changes': [],
            'audio_changes': []
        }), 500


@main_bp.route('/api/toggle_analysis', methods=['POST'])
@main_bp.route('/api/start', methods=['POST'])
def start_analysis():
    try:
        data = request.get_json()
        enabled = data.get('enabled', False)
        
        success = True
        video_success = audio_success = True
        
        if 'video' in InitialConfig.ANALYZER_TYPE:
            video_success = video_analyzer.toggle_analysis(enabled)
            success &= video_success
            if not video_success:
                logger.error("Failed to toggle video analysis")
                
        if 'audio' in InitialConfig.ANALYZER_TYPE:
            # 仅在启用分析时启动音频分析器
            if enabled and not audio_analyzer.is_running:
                try:
                    audio_analyzer.start(audio_analyzer.current_device)
                except Exception as e:
                    logger.error(f"Failed to start audio analyzer before toggling: {str(e)}")
                    audio_success = False
            if audio_analyzer.is_running:
                audio_success = audio_analyzer.toggle_analysis(enabled)
            success &= audio_success
            if not audio_success:
                logger.error("Failed to toggle audio analysis")
                
        if success:
            return jsonify({
                'success': True,
                'analysis_enabled': enabled
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to toggle analyzers'
            }), 500
    except Exception as e:
        logger.error(f"Error in start_analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@main_bp.route('/api/cleanup', methods=['POST'])
def cleanup():
    try:
        if video_analyzer.camera:
            video_analyzer.camera.cleanup()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
