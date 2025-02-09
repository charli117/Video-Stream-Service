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

// 添加初始化历史记录函数
async function initializeHistory() {
    try {
        const response = await fetch('/api/history_files');
        const data = await response.json();
        
        if (data.frame_changes) {
            frameChangesHistory = data.frame_changes;
        }
        
        if (data.audio_changes) {
            audioChangesHistory = data.audio_changes;
        }
        
        // 立即更新日志显示
        updateLogsDisplay();
    } catch (error) {
        console.error('Error initializing history:', error);
    }
}

// 添加统一的日志更新函数
function updateLogsDisplay() {
    // 更新 Video Changes Log
    const videoLogDiv = document.getElementById('video-changes-log');
    if (videoLogDiv) {
        const videoLogHtml = `
            <h3>Video Changes Log</h3>
            <div class="log-entries">
                ${frameChangesHistory.length > 0 ?
                    frameChangesHistory.map(change => {
                        const date = new Date(change.time * 1000);
                        const timeString = date.toLocaleString('zh-CN');
                        return `
                            <div class="log-entry">
                                <a href="javascript:void(0)" onclick="showImageViewer('${change.image_url}')" class="change-time">
                                    <i class="fas fa-camera"></i>
                                    ${timeString}
                                </a>
                            </div>
                        `;
                    }).join('') :
                    '<p class="no-changes">No frame changes detected</p>'
                }
            </div>
        `;
        videoLogDiv.innerHTML = videoLogHtml;
    }

    // 更新 Audio Changes Log
    const audioLogDiv = document.getElementById('audio-changes-log');
    if (audioLogDiv) {
        const audioLogHtml = `
            <h3>Audio Changes Log</h3>
            <div class="log-entries">
                ${audioChangesHistory.length > 0 ?
                    audioChangesHistory.map(change => {
                        const date = new Date(change.time * 1000);
                        const timeString = date.toLocaleString('zh-CN');
                        return `
                            <div class="log-entry">
                                <a href="javascript:void(0)" onclick="playAudioSegment('${change.audio_url}', this)" class="change-time">
                                    <i class="fas fa-volume-up"></i>
                                    ${timeString}
                                </a>
                            </div>
                        `;
                    }).join('') :
                    '<p class="no-changes">No audio changes detected</p>'
                }
            </div>
        `;
        audioLogDiv.innerHTML = audioLogHtml;
    }
}

async function loadDevices() {
    try {
        // 添加加载状态提示
        const statusDiv = document.getElementById('status');
        if (statusDiv) {
            statusDiv.innerHTML = `
                <div class="status-loading">
                    <i class="fas fa-spinner fa-spin"></i> 正在加载摄像头和麦克风...
                </div>
            `;
        }

        const refreshButton = document.getElementById('refreshButton');
        const analysisToggle = document.getElementById('analysisToggle');
        
        // 初始禁用按钮
        refreshButton.disabled = true;
        analysisToggle.disabled = true;
        analysisToggle.title = "正在加载设备列表...";

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

            // 只在本地模式下处理音频设备
            if (window.CAMERA_TYPE === 'local') {
                audioDevices.forEach((device, index) => {
                    audioDeviceNames[index] = device.label || `Audio Device ${index}`;
                });
            }

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

        // 如果是流模式，则触发重连
        if (window.CAMERA_TYPE === 'stream') {
            await fetch("/api/devices/switch", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    camera_index: 0,  // 流模式固定使用索引0
                    audio_index: 0
                })
            });
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
                if (control === 'Refresh Devices' || control === 'Open Analysis') {
                    elements[control].style.display = 'block';
                } else {
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
        while (cameraSelect.options.length > 0) {
            cameraSelect.remove(0);
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
        while (audioSelect.options.length > 0) {
            audioSelect.remove(0);
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

        while (audioSelect.options.length > 0) {
            audioSelect.remove(0);
        }

        // 添加音频设备选项
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
            const option = document.createElement('option');
            option.value = "";
            option.text = "No audio devices found";
            audioSelect.appendChild(option);
        }

        // 恢复之前选中的音频设备（如果有）
        if (currentAudioValue) {
            audioSelect.value = currentAudioValue;
        }
        
        // 绑定事件监听
        if (data.controls.includes('audioSelect') && !data.controls.includes('Switch Devices')) {
            audioSelect.onchange = () => switchDevices();
        }

        // 在设备列表更新完成后检查设备状态
        if (data.cameras?.length > 0 || data.audioDevices?.length > 0) {
            // 获取设备初始化状态
            const statusResponse = await fetch("/api/status");
            const statusData = await statusResponse.json();
            
            if (statusData.devices_ready) {
                analysisToggle.disabled = false;
                analysisToggle.title = "";
            } else {
                analysisToggle.disabled = true;
                analysisToggle.title = "设备未就绪，请等待初始化完成";
            }
        } else {
            analysisToggle.disabled = true;
            analysisToggle.title = "未检测到可用设备";
        }

        // 5. 更新当前设备状态
        currentCamera = data.currentCamera;
        currentAudioDevice = data.currentAudioDevice;
        
        hideError();
        
        // 获取设备状态
        const statusResponse = await fetch("/api/status");
        const statusData = await statusResponse.json();
        
        // 根据设备状态更新按钮
        if (statusData.devices_ready && !statusData.audio_error) {
            analysisToggle.disabled = false;
            analysisToggle.title = "";
            
            // 更新按钮状态
            analysisToggle.setAttribute('data-enabled', statusData.analysis_enabled);
            analysisToggle.innerHTML = statusData.analysis_enabled 
                ? '<i class="fas fa-stop"></i> Close Analysis'
                : '<i class="fas fa-play"></i> Open Analysis';
            updateButtonColor();
        } else {
            analysisToggle.disabled = true;
            analysisToggle.title = statusData.audio_error 
                ? statusData.audio_error.message 
                : "设备未就绪";
        }

    } catch (error) {
        showError('Error loading devices: ' + error.message);
        const analysisToggle = document.getElementById('analysisToggle');
        if (analysisToggle) {
            analysisToggle.disabled = true;
            analysisToggle.title = "设备加载失败";
        }
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
document.addEventListener('DOMContentLoaded', async () => {
    await initializeHistory();  // 首先初始化历史记录
    loadDevices();
    startStatusUpdates();
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

function playAudioSegment(audioUrl, element) {
    // 移除其他播放项的状态
    document.querySelectorAll('.log-entry.playing').forEach(entry => {
        entry.classList.remove('playing');
    });

    // 为当前项添加播放状态
    const currentEntry = element.closest('.log-entry');
    currentEntry.classList.add('playing');

    // 创建音频对象
    const audio = new Audio(audioUrl);

    // 添加时间戳属性用于状态恢复
    const timeAttr = element.getAttribute('data-time');
    if (timeAttr) {
        currentEntry.setAttribute('data-playing-time', timeAttr);
    }

    // 播放结束时移除状态
    audio.onended = () => {
        currentEntry.classList.remove('playing');
    };

    // 错误处理
    audio.onerror = () => {
        currentEntry.classList.remove('playing');
        showError('音频播放失败');
    };

    // 开始播放
    audio.play().catch(error => {
        currentEntry.classList.remove('playing');
        showError('音频播放失败: ' + error.message);
    });
}

// 定期更新状态
let statusUpdateInterval = null;

function startStatusUpdates() {
    updateStatus();  // 立即执行一次
    statusUpdateInterval = setInterval(updateStatus, 1000);  // 每秒更新一次
}

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
    let isEnabled = analysisToggle.getAttribute('data-enabled') === 'true';
    
    // 移除按钮禁用状态
    // analysisToggle.disabled = true; // 删除此行

    if (toggleAnalysisTimeout) {
        clearTimeout(toggleAnalysisTimeout);
    }
    
    // 立即发送请求,不再使用延迟
    fetch('/api/toggle_analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enabled: !isEnabled })
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
            analysisToggle.setAttribute('data-enabled', data.analysis_enabled);
            analysisToggle.innerHTML = data.analysis_enabled 
                ? '<i class="fas fa-stop"></i> Close Analysis'
                : '<i class="fas fa-play"></i> Open Analysis';
            updateButtonColor();
            saveButtonState();
            hideError();
        } else {
            throw new Error(data.error || 'Failed to toggle analysis');
        }
    })
    .catch(error => {
        showError('Error toggling analysis: ' + error.message);
        analysisToggle.innerHTML = isEnabled 
            ? '<i class="fas fa-stop"></i> Close Analysis'
            : '<i class="fas fa-play"></i> Open Analysis';
        updateButtonColor(); 
    });
}

// 更新按钮颜色的函数
function updateButtonColor() {
    const analysisToggle = document.getElementById('analysisToggle');
    if (analysisToggle.textContent.trim() === 'Close Analysis') {
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

// 修改 updateStatus 函数中的相关部分
async function updateStatus() {
    try {
        const response = await fetch("/api/status");
        const status = await response.json();
        const analysisToggle = document.getElementById('analysisToggle');
        
        if (analysisToggle) {
            // 根据设备状态更新分析按钮，但不影响日志显示
            if (!status.devices_ready || !status.camera_info.initialized) {
                analysisToggle.disabled = true;
                analysisToggle.title = "设备未就绪，请等待初始化完成";
            } else if (status.audio_error) {
                analysisToggle.disabled = true;
                analysisToggle.title = status.audio_error.message;
            } else {
                analysisToggle.disabled = false;
                analysisToggle.title = "";
                analysisToggle.setAttribute('data-enabled', status.analysis_enabled);
                analysisToggle.innerHTML = status.analysis_enabled 
                    ? '<i class="fas fa-stop"></i> Close Analysis'
                    : '<i class="fas fa-play"></i> Open Analysis';
                updateButtonColor();
            }
        }
        
        // 更新基本状态信息
        const statusDiv = document.getElementById('status');
        if (!statusDiv) return;
        let statusHtml = '';

        if (status.error) {
            statusHtml = `<p class="error">Error: ${status.error}</p>`;
            statusDiv.innerHTML = statusHtml;
            return;
        }
        
        statusHtml = `
            <p><i class="fas fa-circle ${status.is_running ? 'text-success' : 'text-danger'}"></i> Status: ${status.is_running ? 'Running' : 'Stopped'}</p>
            <p><i class="fas fa-video"></i> Camera: ${status.current_camera_name || 'None'}</p>
            <p><i class="fas fa-microphone"></i> Audio: ${status.current_audio_name || 'None'}</p>
            <p><i class="fas fa-expand"></i> Resolution: ${status.camera_info.width || 0}x${status.camera_info.height || 0}</p>
            <p><i class="fas fa-tachometer-alt"></i> FPS: ${status.fps || 0}</p>
        `;
        
        if (!status.devices_ready) {
            if (analysisToggle) {
                analysisToggle.disabled = true;
                analysisToggle.title = "设备未就绪";
            }
            statusHtml += `
                <div class="warning">
                    <i class="fas fa-exclamation-circle"></i>设备未就绪，请等待初始化完成
                </div>
            `;
        } else if (status.audio_error) {
            if (analysisToggle) {
                analysisToggle.disabled = true;
                analysisToggle.title = status.audio_error.message;
            }
            statusHtml += `
                <div class="error">
                    <i class="fas fa-exclamation-triangle"></i>
                    音频错误: ${status.audio_error.message}
                </div>
            `;
        } else {
            if (analysisToggle) {
                analysisToggle.disabled = false;
                analysisToggle.title = "";
            }
        }
        
        statusDiv.innerHTML = statusHtml;

        // 更新历史记录（只追加不重复，最多保存50条）
        let historyUpdated = false;

        if (status.frame_changes?.length > 0) {
            status.frame_changes.forEach(change => {
                if (!frameChangesHistory.find(h => h.time === change.time)) {
                    frameChangesHistory.unshift(change);
                    historyUpdated = true;
                }
            });
            frameChangesHistory = frameChangesHistory.slice(0, 50);
        }

        if (status.audio_changes?.length > 0) {
            status.audio_changes.forEach(change => {
                if (!audioChangesHistory.find(h => h.time === change.time)) {
                    audioChangesHistory.unshift(change);
                    historyUpdated = true;
                }
            });
            audioChangesHistory = audioChangesHistory.slice(0, 50);
        }

        // 独立更新 Video Changes Log
        const videoLogDiv = document.getElementById('video-changes-log');
        if (videoLogDiv) {
            const videoLogHtml = `
                <h3>Video Changes Log</h3>
                <div class="log-entries">
                    ${frameChangesHistory.length > 0 ?
                        frameChangesHistory.map(change => {
                            const date = new Date(change.time * 1000);
                            const timeString = date.toLocaleString('zh-CN');
                            return `
                                <div class="log-entry">
                                    <a href="javascript:void(0)" onclick="showImageViewer('${change.image_url}')" class="change-time">
                                        <i class="fas fa-camera"></i>
                                        ${timeString}
                                    </a>
                                </div>
                            `;
                        }).join('') :
                        '<p class="no-changes">No frame changes detected</p>'
                    }
                </div>
            `;
            videoLogDiv.innerHTML = videoLogHtml;
        }

        // 独立更新 Audio Changes Log
        const audioLogDiv = document.getElementById('audio-changes-log');
        if (audioLogDiv) {
            const currentPlayingEntry = document.querySelector('.log-entry.playing');
            const currentPlayingTime = currentPlayingEntry ? 
                currentPlayingEntry.querySelector('.change-time').getAttribute('data-time') : null;

            const audioLogHtml = `
                <h3>Audio Changes Log</h3>
                <div class="log-entries">
                    ${audioChangesHistory.length > 0 ?
                        audioChangesHistory.map(change => {
                            const date = new Date(change.time * 1000);
                            const timeString = date.toLocaleString('zh-CN');
                            return `
                                <div class="log-entry ${change.time === currentPlayingTime ? 'playing' : ''}">
                                    <a href="javascript:void(0)" 
                                       onclick="playAudioSegment('${change.audio_url}', this)"
                                       class="change-time"
                                       data-time="${change.time}">
                                        <i class="fas fa-volume-up"></i>
                                        ${timeString}
                                    </a>
                                </div>
                            `;
                        }).join('') :
                        '<p class="no-changes">No audio changes detected</p>'
                    }
                </div>
            `;
            audioLogDiv.innerHTML = audioLogHtml;
        }

        // 在 updateStatus 函数中更新日志部分的代码
        const frameLogEntries = document.getElementById('frameLogEntries');
        if (frameLogEntries) {
            frameLogEntries.innerHTML = frameChangesHistory.length > 0 ?
                frameChangesHistory.map(change => {
                    const timeString = change.time ? new Date(change.time * 1000).toLocaleString('zh-CN') : 'N/A';
                    return `
                        <div class="log-entry">
                            <a href="javascript:void(0)" onclick="showImageViewer('${change.image_url}')" class="change-time">
                                <i class="fas fa-camera"></i>
                                ${timeString}
                            </a>
                        </div>
                    `;
                }).join('') :
                '<p class="no-changes">No frame changes detected</p>';
        }

        const audioLogEntries = document.getElementById('audioLogEntries');
        if (audioLogEntries) {
            audioLogEntries.innerHTML = audioChangesHistory.length > 0 ?
                audioChangesHistory.map(change => {
                    const timeString = change.time ? new Date(change.time * 1000).toLocaleString('zh-CN') : 'N/A';
                    return `
                        <div class="log-entry">
                            <a href="javascript:void(0)" onclick="playAudioSegment('${change.audio_url}', this)" class="change-time">
                                <i class="fas fa-volume-up"></i>
                                ${timeString}
                            </a>
                        </div>
                    `;
                }).join('') :
                '<p class="no-changes">No audio changes detected</p>';
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
