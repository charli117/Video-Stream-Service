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
        refreshButton.disabled = true;

        // 1. 获取设备名称并发送到服务器
        let deviceNames = {};
        let audioDeviceNames = {};
        
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            const audioDevices = devices.filter(device => device.kind === 'audioinput');
            
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
        }

        // 2. 获取设备列表和控件配置
        const response = await fetch("/api/devices");
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        // 3. 更新控件显示状态
        const controls = data.controls || [];
        const elements = {
            'cameraSelect': document.getElementById('cameraSelect').parentElement,
            'audioSelect': document.getElementById('audioSelect').parentElement,
            'Refresh Devices': document.getElementById('refreshButton').parentElement,
            'Open Analysis': document.getElementById('analysisToggle').parentElement
        };
        
        // 更新控件显示状态
        Object.keys(elements).forEach(control => {
            if (elements[control]) {
                // 始终显示 Refresh Devices 和 Open Analysis 按钮
                if (control === 'Refresh Devices' || control === 'Open Analysis') {
                    elements[control].style.display = 'block';
                } else {
                    // 其他控件根据 controls 数组决定是否显示
                    elements[control].style.display = controls.includes(control) ? 'block' : 'none';
                }
            }
        });

        // 4. 更新设备选择列表
        const cameraSelect = document.getElementById('cameraSelect');
        const audioSelect = document.getElementById('audioSelect');
        
        // 更新摄像头选择
        // 保存当前选中值
        const currentCameraValue = cameraSelect.value;
        
        // 清空选项但保留第一个默认选项
        while (cameraSelect.options.length > 1) {
            cameraSelect.remove(1);
        }
        
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
            
            // 恢复选中值
            if (currentCameraValue) {
                cameraSelect.value = currentCameraValue;
            }
            
            if (data.controls.includes('cameraSelect') && !data.controls.includes('Switch Devices')) {
                cameraSelect.onchange = () => switchDevices();
            }
        }

        // 更新音频设备选择
        // 保存当前选中值
        const currentAudioValue = audioSelect.value;
        
        // 清空选项但保留第一个默认选项
        while (audioSelect.options.length > 1) {
            audioSelect.remove(1);
        }
        
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
            
            // 恢复选中值
            if (currentAudioValue) {
                audioSelect.value = currentAudioValue;
            }
            
            if (data.controls.includes('audioSelect') && !data.controls.includes('Switch Devices')) {
                audioSelect.onchange = () => switchDevices();
            }
        }

        // 重新初始化 select 元素的样式
        if (typeof $ !== 'undefined') {
            $(cameraSelect).selectpicker('refresh');
            $(audioSelect).selectpicker('refresh');
        }

        // 为本地摄像头模式添加change事件监听
        if (data.controls.includes('cameraSelect') && !data.controls.includes('Switch Devices')) {
            cameraSelect.onchange = () => switchDevices();
        } else {
            cameraSelect.innerHTML = '<option value="">No cameras found</option>';
        }

        // 更新音频设备选择
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
            
            // 为本地音频设备模式添加change事件监听
            if (data.controls.includes('audioSelect') && !data.controls.includes('Switch Devices')) {
                audioSelect.onchange = () => switchDevices();
            }
        } else {
            audioSelect.innerHTML = '<option value="">No audio devices found</option>';
        }

        // 5. 更新当前设备状态
        currentCamera = data.currentCamera;
        currentAudioDevice = data.currentAudioDevice;
        
        hideError();
        
    } catch (error) {
        showError('Error loading devices: ' + error.message);
    } finally {
        refreshButton.disabled = false;
    }
}

// 如果需要切换设备，直接调用 switchDevices
async function onDeviceChange() {
    const cameraSelect = document.getElementById('cameraSelect');
    const audioSelect = document.getElementById('audioSelect');
    
    if (cameraSelect && audioSelect) {
        await switchDevices();
    }
}

// 为设备选择添加 change 事件监听器
document.addEventListener('DOMContentLoaded', () => {
    const cameraSelect = document.getElementById('cameraSelect');
    const audioSelect = document.getElementById('audioSelect');
    
    if (cameraSelect) {
        cameraSelect.addEventListener('change', onDeviceChange);
    }
    
    if (audioSelect) {
        audioSelect.addEventListener('change', onDeviceChange);
    }
});

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
        // 只禁用刷新按钮
        const refreshButton = document.getElementById('refreshButton');
        if (refreshButton) {
            refreshButton.disabled = true;
        }

        console.log(`Switching to camera ${cameraIndex} and audio device ${audioIndex}`);

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

        console.log(`Switched to camera ${cameraIndex} and audio device ${audioIndex}`);

        // 刷新视频源
        const videoFeed = document.getElementById('videoFeed');
        videoFeed.src = "/video_feed?" + new Date().getTime();

        hideError();

    } catch (error) {
        showError('Error switching devices: ' + error.message);
    } finally {
        // 只恢复刷新按钮
        const refreshButton = document.getElementById('refreshButton');
        if (refreshButton) {
            refreshButton.disabled = false;
        }
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

        const response = await fetch("/api/devices/switch", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                camera_index: cameraIndex,
                audio_index: currentAudioDevice // 使用全局变量 currentAudioDevice
            })
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
let statusUpdateInterval = null;

function startStatusUpdates() {
    // 如果已经存在定时器，先清除
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
    
    // 立即更新一次
    updateStatus().then();
    
    // 设置新的定时器
    statusUpdateInterval = setInterval(updateStatus, 500);
}

// 页面加载完成后启动状态更新
document.addEventListener('DOMContentLoaded', () => {
    loadDevices();
    startStatusUpdates();
    updateButtonColor();
});

// 只保留这一个事件监听器
document.addEventListener('DOMContentLoaded', () => {
    loadDevices();  // 使用新的 loadDevices 替换 loadCameras
    startStatusUpdates();
    updateButtonColor();
});

window.addEventListener('beforeunload', async function(e) {
    try {
        // 在页面刷新前尝试清理资源
        await fetch('/api/cleanup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
    } catch (error) {
        console.error('Cleanup error:', error);
    }
});

// 添加防抖函数
let toggleAnalysisTimeout;

function toggleAnalysis() {
    const analysisToggle = document.getElementById('analysisToggle');
    const isEnabled = analysisToggle.textContent === 'Open Analysis';
    
    // 禁用按钮防止重复点击
    analysisToggle.disabled = true;
    
    // 清除之前的定时器
    if (toggleAnalysisTimeout) {
        clearTimeout(toggleAnalysisTimeout);
    }
    
    // 设置新的定时器
    toggleAnalysisTimeout = setTimeout(() => {
        fetch('/api/toggle_analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ enabled: isEnabled })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to toggle analysis');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                analysisToggle.textContent = data.analysis_enabled ? 'Close Analysis' : 'Open Analysis';
                updateButtonColor();
                saveButtonState();
                hideError();  // 成功时隐藏错误信息
            } else {
                throw new Error(data.error || 'Failed to toggle analysis');
            }
        })
        .catch(error => {
            showError('Error toggling analysis: ' + error.message);
            // 发生错误时恢复按钮状态
            analysisToggle.textContent = isEnabled ? 'Close Analysis' : 'Open Analysis';
            updateButtonColor();
        })
        .finally(() => {
            // 重新启用按钮
            analysisToggle.disabled = false;
        });
    }, 300);
}

// 更新按钮颜色的函数
function updateButtonColor() {
    const analysisToggle = document.getElementById('analysisToggle');
    const icon = document.createElement('i');
    if (analysisToggle.textContent === 'Close Analysis') {
        analysisToggle.style.backgroundColor = 'red';
        icon.className = 'fas fa-stop';
        icon.style.marginRight = '5px';
        icon.style.display = 'inline-block';
    } else {
        analysisToggle.style.backgroundColor = '';
        icon.className = 'fas fa-play';
        icon.style.marginRight = '5px';
        icon.style.display = 'inline-block';
    }
    analysisToggle.appendChild(icon);
}

// 保存按钮状态到 localStorage
function saveButtonState() {
    const analysisToggle = document.getElementById('analysisToggle');
    localStorage.setItem('analysisToggleState', analysisToggle.textContent);
}

// 修改 updateStatus 函数中的相关部分
async function updateStatus() {
    try {
        const response = await fetch("/api/status");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const status = await response.json();

        // 更新分析按钮状态时增加判断
        const analysisToggle = document.getElementById('analysisToggle');
        if (analysisToggle) {
            const currentState = analysisToggle.textContent === 'Close Analysis';
            // 只在状态确实发生变化时才更新
            if (currentState !== status.analysis_enabled) {
                analysisToggle.textContent = status.analysis_enabled ? 'Close Analysis' : 'Open Analysis';
                updateButtonColor();
                saveButtonState();
            }
        }

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
