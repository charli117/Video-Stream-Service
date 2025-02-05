let currentCamera = null;
let frameChangesHistory = [];  // 存储历史记录
let audioChangesHistory = [];  // 存储音频历史记录

// 添加图片查看相关函数
function showImageViewer(imageUrl) {
    const viewer = document.createElement('div');
    viewer.className = 'image-viewer';
    viewer.innerHTML = `
        <div class="viewer-content">
            <div class="viewer-header">
                <button class="zoom-in"><i class="fas fa-search-plus"></i></button>
                <button class="zoom-out"><i class="fas fa-search-minus"></i></button>
                <button class="close-viewer"><i class="fas fa-times"></i></button>
            </div>
            <div class="viewer-body">
                <img src="${imageUrl}" alt="Frame Change">
            </div>
        </div>
    `;

    document.body.appendChild(viewer);

    // 绑定事件
    const img = viewer.querySelector('img');
    let scale = 1;

    viewer.querySelector('.zoom-in').onclick = () => {
        scale *= 1.2;
        img.style.transform = `scale(${scale})`;
    };

    viewer.querySelector('.zoom-out').onclick = () => {
        scale /= 1.2;
        img.style.transform = `scale(${scale})`;
    };

    viewer.querySelector('.close-viewer').onclick = () => {
        viewer.remove();
    };

    // 点击背景关闭
    viewer.onclick = (e) => {
        if (e.target === viewer) {
            viewer.remove();
        }
    };
}

async function loadDevices() {
    try {
        const refreshButton = document.getElementById('refreshButton');
        const switchButton = document.getElementById('switchButton');

        refreshButton.disabled = true;
        switchButton.disabled = true;

        // 获取设备名称
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            const audioDevices = devices.filter(device => device.kind === 'audioinput');
            
            const deviceNames = {};
            const audioDeviceNames = {};

            videoDevices.forEach((device, index) => {
                deviceNames[index] = device.label || `Camera ${index}`;
            });

            audioDevices.forEach((device, index) => {
                audioDeviceNames[index] = device.label || `Audio Device ${index}`;
            });

            // 发送设备名称到服务器
            await fetch('/api/device_names', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    deviceNames,
                    audioDeviceNames
                })
            });

            
            videoDevices.forEach((device, index) => {
                deviceNames.video[index] = device.label || `Camera ${index}`;
            });
            
            audioDevices.forEach((device, index) => {
                deviceNames.audio[index] = device.label || `Microphone ${index}`;
            });

            // 发送设备名称到服务器
            await fetch('/api/device_names', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ deviceNames })
            });
        }

        // 获取设备列表
        const response = await fetch("/api/devices");
        const data = await response.json();

        // 更新摄像头选择
        const cameraSelect = document.getElementById('cameraSelect');
        cameraSelect.innerHTML = '';
        
        if (data.cameras && data.cameras.length > 0) {
            data.cameras.forEach(camera => {
                const option = document.createElement('option');
                option.value = camera.index;
                option.text = `${camera.name} (${camera.width}x${camera.height} @ ${camera.fps}fps)`;
                if (camera.index === data.currentCamera) {
                    option.selected = true;
                }
                cameraSelect.appendChild(option);
            });
        } else {
            cameraSelect.innerHTML = '<option value="">No cameras found</option>';
        }

        // 更新音频设备选择
        const audioSelect = document.getElementById('audioSelect');
        audioSelect.innerHTML = '';
        
        if (data.audioDevices && data.audioDevices.length > 0) {
            data.audioDevices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.index;
                option.text = device.name;
                if (device.index === data.currentAudioDevice) {
                    option.selected = true;
                }
                audioSelect.appendChild(option);
            });
        } else {
            audioSelect.innerHTML = '<option value="">No audio devices found</option>';
        }

        currentCamera = data.currentCamera;
        currentAudioDevice = data.currentAudioDevice;
        switchButton.disabled = false;

    } catch (error) {
        showError('Error loading devices: ' + error.message);
    } finally {
        refreshButton.disabled = false;
    }
}

async function switchDevices() {
    const cameraSelect = document.getElementById('cameraSelect');
    const audioSelect = document.getElementById('audioSelect');
    const cameraIndex = parseInt(cameraSelect.value);
    const audioIndex = parseInt(audioSelect.value);

    if (isNaN(cameraIndex) || isNaN(audioIndex)) {
        showError('Please select both camera and audio device');
        return;
    }

    try {
        document.getElementById('switchButton').disabled = true;
        document.getElementById('refreshButton').disabled = true;

        const response = await fetch("/api/devices/switch", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                camera_index: cameraIndex,
                audio_index: audioIndex
            })
        });

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || 'Failed to switch devices');
        }

        currentCamera = cameraIndex;
        currentAudioDevice = audioIndex;

        // 刷新视频源
        const videoFeed = document.getElementById('videoFeed');
        videoFeed.src = "/video_feed?" + new Date().getTime();

        hideError();

    } catch (error) {
        showError('Error switching devices: ' + error.message);
    } finally {
        document.getElementById('switchButton').disabled = false;
        document.getElementById('refreshButton').disabled = false;
    }
}

// 替换原有的 loadCameras 函数调用
document.addEventListener('DOMContentLoaded', () => {
    loadDevices();
    startStatusUpdates();
    restoreButtonState();
    updateButtonColor();
});

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

function playAudioSegment(audioUrl) {
    const audio = new Audio(audioUrl);
    audio.play();
}

// 定期更新状态
function startStatusUpdates() {
    // 立即更新一次
    updateStatus().then();
    // 更频繁地更新状态（每200ms一次）以便及时显示帧变化
    setInterval(updateStatus, 200);
}

// 页面加载完成后启动状态更新
document.addEventListener('DOMContentLoaded', () => {
    loadCameras();
    startStatusUpdates();
    restoreButtonState();  // 添加这一行
    updateButtonColor();
});

function toggleAnalysis() {
    const analysisToggle = document.getElementById('analysisToggle');
    const isEnabled = analysisToggle.textContent === 'Open Analysis';

    fetch('/api/toggle_analysis', {
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

// 更新按钮颜色的函数
function updateButtonColor() {
    const analysisToggle = document.getElementById('analysisToggle');
    if (analysisToggle.textContent === 'Close Analysis') {
        analysisToggle.style.backgroundColor = 'red';
    } else {
        analysisToggle.style.backgroundColor = '';
    }
}

// 保存按钮状态到 localStorage
function saveButtonState() {
    const analysisToggle = document.getElementById('analysisToggle');
    localStorage.setItem('analysisToggleState', analysisToggle.textContent);
}

// 修改现有的 restoreButtonState 函数
function restoreButtonState() {
    const analysisToggle = document.getElementById('analysisToggle');
    const savedAnalysisState = localStorage.getItem('analysisToggleState');
    if (savedAnalysisState) {
        analysisToggle.textContent = savedAnalysisState;
        updateButtonColor();
    }
}

// 修改 updateStatus 函数中的相关部分
async function updateStatus() {
    try {
        const response = await fetch("/api/status");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const status = await response.json();

        // 更新帧变化历史记录
        if (status.frame_changes && status.frame_changes.length > 0) {
            status.frame_changes.forEach(change => {
                const existingChange = frameChangesHistory.find(h => h.time === change.time);
                if (!existingChange) {
                    frameChangesHistory.unshift(change);
                }
            });
            frameChangesHistory = frameChangesHistory.slice(0, 50);
        }

        // 更新音频变化历史记录
        if (status.audio_changes && status.audio_changes.length > 0) {
            status.audio_changes.forEach(change => {
                const existingChange = audioChangesHistory.find(h => h.time === change.time);
                if (!existingChange) {
                    audioChangesHistory.unshift(change);
                }
            });
            audioChangesHistory = audioChangesHistory.slice(0, 50);
        }

        // 更新状态显示
        const statusDiv = document.getElementById('status');
        const analysisToggle = document.getElementById('analysisToggle');

        if (status.error) {
            statusDiv.innerHTML = `
                <p class="error">Error: ${status.error}</p>
            `;
            return;
        }

        // 更新分析按钮状态
        if (analysisToggle) {
            analysisToggle.textContent = status.analysis_enabled ? 'Close Analysis' : 'Open Analysis';
            updateButtonColor();
        }

        let statusHtml = `
            <p>Status: ${status.is_running ? 'Running' : 'Stopped'}</p>
            <p>Camera: ${status.current_camera_name || 'None'}</p>
            <p>Audio: ${status.current_audio_name || 'None'}</p>
            <p>Resolution: ${status.camera_info.width || 0}x${status.camera_info.height || 0}</p>
            <p>FPS: ${status.fps || 0}</p>
            <p>Analysis: ${status.analysis_enabled ? 'Enabled' : 'Disabled'}</p>
        `;

        // 始终显示帧变化日志，移除 if (status.analysis_enabled) 判断
        statusHtml += `
            <div class="changes-log">
                <h3>Video Changes Log</h3>
                <div class="log-entries">
        `;
        
        if (frameChangesHistory.length > 0) {
            frameChangesHistory.forEach(change => {
                const date = new Date(change.time * 1000);
                const timeString = date.toLocaleString('zh-CN');
                statusHtml += `
                    <div class="log-entry">
                        <a href="javascript:void(0)" onclick="showImageViewer('${change.image_url}')" class="change-time">
                            <i class="fas fa-camera"></i>
                            ${timeString}
                        </a>
                    </div>
                `;
            });
        } else {
            statusHtml += `
                <p class="no-changes">No frame changes detected</p>
            `;
        }

        statusHtml += `
                </div>
            </div>
        `;

        // 添加音频变化日志
        statusHtml += `
            <div class="changes-log">
                <h3>Audio Changes Log</h3>
                <div class="log-entries">
        `;
        
        if (audioChangesHistory.length > 0) {
            audioChangesHistory.forEach(change => {
                const date = new Date(change.time * 1000);
                const timeString = date.toLocaleString('zh-CN');
                statusHtml += `
                    <div class="log-entry">
                        <a href="javascript:void(0)" onclick="playAudio('${change.audio_url}')" class="change-time">
                            <i class="fas fa-volume-up"></i>
                            ${timeString}
                        </a>
                    </div>
                `;
            });
        } else {
            statusHtml += `
                <p class="no-changes">No audio changes detected</p>
            `;
        }

        statusHtml += `
                </div>
            </div>
        `;

        statusDiv.innerHTML = statusHtml;

        // 如果摄像头未初始化，显示警告
        if (!status.camera_info.initialized) {
            statusDiv.innerHTML += `
                <p class="warning">Camera not initialized</p>
            `;
        }

        // 更新音频变化记录
        if (status.audio_changes && status.audio_changes.length > 0) {
            const audioLogEntries = document.getElementById('audioLogEntries');
            let audioHtml = '';
            
            status.audio_changes.forEach(change => {
                const date = new Date(change.time * 1000);
                const timeString = date.toLocaleString('zh-CN');
                audioHtml += `
                    <div class="log-entry">
                        <a href="javascript:void(0)" onclick="playAudioSegment('${change.audio_url}')" class="change-time">
                            <i class="fas fa-volume-up"></i>
                            ${timeString}
                        </a>
                    </div>
                `;
            });
            
            audioLogEntries.innerHTML = audioHtml;
        }
        
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// 添加音频播放功能
function playAudio(audioUrl) {
    const audio = new Audio(audioUrl);
    audio.play().catch(error => {
        showError('Error playing audio: ' + error.message);
    });
}
