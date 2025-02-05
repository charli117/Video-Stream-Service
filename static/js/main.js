let currentCamera = null;

async function loadCameras() {
    try {
        const refreshButton = document.getElementById('refreshButton');
        const switchButton = document.getElementById('switchButton');

        // 禁用按钮，显示加载状态
        refreshButton.disabled = true;
        switchButton.disabled = true;

        const response = await fetch("/api/cameras");
        const data = await response.json();

        const select = document.getElementById('cameraSelect');
        select.innerHTML = '';

        // 使用data.cameras而不是直接使用返回数据
        if (!data.cameras || data.cameras.length === 0) {
            select.innerHTML = '<option value="">No cameras found</option>';
            return;
        }

        // 更新当前摄像头
        currentCamera = data.current;

        // 遍历cameras数组
        data.cameras.forEach(camera => {
            const option = document.createElement('option');
            option.value = camera.index;
            option.text = `Camera ${camera.index} (${camera.width}x${camera.height} @ ${camera.fps}fps)`;
            if (camera.index === currentCamera) {
                option.selected = true;
            }
            select.appendChild(option);
        });

        switchButton.disabled = false;

    } catch (error) {
        showError('Error loading cameras: ' + error.message);
    } finally {
        // 重新启用刷新按钮
        document.getElementById('refreshButton').disabled = false;
    }
}

async function switchCamera() {
    const select = document.getElementById('cameraSelect');
    const cameraIndex = parseInt(select.value);

    if (isNaN(cameraIndex)) {
        showError('Please select a camera');
        return;
    }

    try {
        document.getElementById('switchButton').disabled = true;
        document.getElementById('refreshButton').disabled = true;

        const response = await fetch("/api/camera/switch", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({camera_index: cameraIndex})
        });

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || 'Failed to switch camera');
        }

        // 更新当前摄像头
        currentCamera = cameraIndex;

        // 刷新视频源
        const videoFeed = document.getElementById('videoFeed');
        videoFeed.src = "/video_feed?" + new Date().getTime();

        hideError();

    } catch (error) {
        showError('Error switching camera: ' + error.message);
    } finally {
        document.getElementById('switchButton').disabled = false;
        document.getElementById('refreshButton').disabled = false;
    }
}

function showError(message) {
    const error = document.getElementById('error');
    error.textContent = message;
    error.style.display = 'block';
}

function hideError() {
    const error = document.getElementById('error');
    error.style.display = 'none';
}

async function updateStatus() {
    try {
        const response = await fetch("/api/status");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const status = await response.json();

        // 更新状态显示
        const statusDiv = document.getElementById('status');
        if (status.error) {
            statusDiv.innerHTML = `
                <p class="error">Error: ${status.error}</p>
            `;
            return;
        }

        statusDiv.innerHTML = `
                    <p>Status: ${status.is_running ? 'Running' : 'Stopped'}</p>
                    <p>Camera: ${status.current_camera !== null ? status.current_camera : 'None'}</p>
                    <p>Resolution: ${status.camera_info.width || 0}x${status.camera_info.height || 0}</p>
                    <p>FPS: ${status.fps || 0}</p>
                `;

        // 如果摄像头未初始化，显示警告
        if (!status.camera_info.initialized) {
            statusDiv.innerHTML += `
                        <p class="warning">Camera not initialized</p>
                    `;
        }

    } catch (error) {
        console.error('Error updating status:', error);
        const statusDiv = document.getElementById('status');
        statusDiv.innerHTML = `
                    <p class="error">Error updating status: ${error.message}</p>
                `;
    }
}

// 定期更新状态
function startStatusUpdates() {
    // 立即更新一次
    updateStatus();
    // 然后每秒更新一次
    setInterval(updateStatus, 1000);
}

// 页面加载完成后启动状态更新
document.addEventListener('DOMContentLoaded', () => {
    loadCameras();
    startStatusUpdates();
    restoreButtonState();
});

function toggleAnalysis() {
    const analysisToggle = document.getElementById('analysisToggle');
    const isEnabled = analysisToggle.textContent === 'Open Analysis';

    fetch('/toggle_analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enabled: isEnabled })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            analysisToggle.textContent = data.analysis_enabled ? 'Close Analysis' : 'Open Analysis';
            updateButtonColor();
            saveButtonState();
        } else {
            showError('Failed to toggle analysis');
        }
    })
    .catch(error => {
        showError('Error toggling analysis: ' + error.message);
    });
}

function updateButtonColor() {
    const analysisToggle = document.getElementById('analysisToggle');
    if (analysisToggle.textContent === 'Close Analysis') {
        analysisToggle.style.backgroundColor = 'red';
    } else {
        analysisToggle.style.backgroundColor = '';
    }
}

function saveButtonState() {
    const analysisToggle = document.getElementById('analysisToggle');
    localStorage.setItem('analysisToggleState', analysisToggle.textContent);
}

function restoreButtonState() {
    const analysisToggle = document.getElementById('analysisToggle');
    const savedState = localStorage.getItem('analysisToggleState');
    if (savedState) {
        analysisToggle.textContent = savedState;
        updateButtonColor();
    }
}
