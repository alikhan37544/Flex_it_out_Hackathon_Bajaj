/**
 * FitTrackAI Sit-up Counter
 * Advanced pose detection and exercise counting application
 */

// Global variables for application state
const state = {
    // Core application state
    camera: {
        active: false,
        stream: null,
        width: 640,
        height: 480
    },
    tracking: {
        active: false,
        model: null,
        lastPose: null,
        confidence: 0.5
    },
    situp: {
        count: 0,
        inUpPosition: false,
        inDownPosition: true,
        angleTorsoLeg: 0,
        threshold: 25, // Default threshold angle in degrees
        tempCount: 0,
        previousAngles: [] // Store previous angles for smoothing
    },
    ui: {
        activeSection: 'workout',
        showSkeleton: true,
        showAngles: true,
        mirrorMode: true,
        audioFeedback: true,
        voiceGuide: false
    },
    stats: {
        today: 0,
        weekly: [0, 0, 0, 0, 0, 0, 0], // Last 7 days
        best: 0,
        chart: null
    }
};

// DOM Elements
const elements = {
    // Main sections
    sections: {
        workout: document.getElementById('workout'),
        stats: document.getElementById('stats'),
        settings: document.getElementById('settings')
    },
    // Nav buttons
    navButtons: document.querySelectorAll('.nav-btn'),
    
    // Camera & tracking
    video: document.getElementById('video'),
    canvas: document.getElementById('output'),
    startCameraBtn: document.getElementById('start-camera'),
    startTrackingBtn: document.getElementById('start-tracking'),
    resetCounterBtn: document.getElementById('reset-counter'),
    
    // Counters & indicators
    situpCount: document.getElementById('sit-up-count'),
    repIndicator: document.getElementById('rep-indicator'),
    repStatus: document.querySelector('.rep-status'),
    progressBar: document.querySelector('.progress-bar'),
    
    // Stats elements
    todayCount: document.getElementById('today-count'),
    weeklyCount: document.getElementById('weekly-count'),
    bestCount: document.getElementById('best-count'),
    weeklyChart: document.getElementById('weekly-chart'),
    
    // Settings elements
    detectionConfidence: document.getElementById('detection-confidence'),
    confidenceValue: document.getElementById('confidence-value'),
    repThreshold: document.getElementById('rep-threshold'),
    thresholdValue: document.getElementById('threshold-value'),
    showSkeleton: document.getElementById('show-skeleton'),
    showAngles: document.getElementById('show-angles'),
    mirrorMode: document.getElementById('mirror-mode'),
    audioFeedback: document.getElementById('audio-feedback'),
    voiceGuide: document.getElementById('voice-guide'),
    
    // Modal and overlay
    permissionModal: document.getElementById('permission-modal'),
    grantPermissionBtn: document.getElementById('grant-permission'),
    closeModalBtn: document.querySelector('.close-modal'),
    loadingOverlay: document.getElementById('loading-overlay'),
    
    // Audio
    repCompleteSound: document.getElementById('rep-complete-sound'),
    milestoneSound: document.getElementById('milestone-sound')
};

// Drawing constants
const POSE_CONNECTIONS = [
    ['leftShoulder', 'rightShoulder'], 
    ['leftShoulder', 'leftHip'], 
    ['rightShoulder', 'rightHip'], 
    ['leftHip', 'rightHip'],
    ['leftShoulder', 'leftElbow'], 
    ['leftElbow', 'leftWrist'],
    ['rightShoulder', 'rightElbow'], 
    ['rightElbow', 'rightWrist'],
    ['leftHip', 'leftKnee'], 
    ['leftKnee', 'leftAnkle'],
    ['rightHip', 'rightKnee'], 
    ['rightKnee', 'rightAnkle']
];

// Speech synthesis for voice guidance
const speechSynthesis = window.speechSynthesis;

/**
 * Application Initialization
 */
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadSettings();
    loadStats();
    initStatsChart();
});

/**
 * Initialize all event listeners
 */
function initEventListeners() {
    // Navigation
    elements.navButtons.forEach(button => {
        button.addEventListener('click', () => switchSection(button.dataset.section));
    });
    
    // Camera controls
    elements.startCameraBtn.addEventListener('click', setupCamera);
    elements.startTrackingBtn.addEventListener('click', toggleTracking);
    elements.resetCounterBtn.addEventListener('click', resetCounter);
    
    // Modal interactions
    elements.grantPermissionBtn.addEventListener('click', setupCamera);
    elements.closeModalBtn.addEventListener('click', () => {
        elements.permissionModal.style.display = 'none';
    });
    
    // Settings changes
    elements.detectionConfidence.addEventListener('input', updateConfidence);
    elements.repThreshold.addEventListener('input', updateThreshold);
    elements.showSkeleton.addEventListener('change', updateSettingsBool);
    elements.showAngles.addEventListener('change', updateSettingsBool);
    elements.mirrorMode.addEventListener('change', updateSettingsBool);
    elements.audioFeedback.addEventListener('change', updateSettingsBool);
    elements.voiceGuide.addEventListener('change', updateSettingsBool);
    
    // Handle clicks outside the modal to close it
    window.addEventListener('click', (event) => {
        if (event.target === elements.permissionModal) {
            elements.permissionModal.style.display = 'none';
        }
    });
}

/**
 * Switch between app sections (workout, stats, settings)
 */
function switchSection(sectionId) {
    // Hide all sections
    Object.values(elements.sections).forEach(section => {
        section.classList.remove('active-section');
    });
    
    // Show selected section
    elements.sections[sectionId].classList.add('active-section');
    
    // Update nav button state
    elements.navButtons.forEach(button => {
        if (button.dataset.section === sectionId) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
    
    // Update current section in state
    state.ui.activeSection = sectionId;
    
    // Refresh charts if switching to stats
    if (sectionId === 'stats' && state.stats.chart) {
        state.stats.chart.update();
    }
}

/**
 * Camera Setup & Permissions
 */
async function setupCamera() {
    elements.permissionModal.style.display = 'none';
    elements.loadingOverlay.style.display = 'flex';
    
    try {
        // Request camera access with constraints
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: 'user',
                width: { ideal: state.camera.width },
                height: { ideal: state.camera.height }
            }
        });
        
        elements.video.srcObject = stream;
        state.camera.stream = stream;
        state.camera.active = true;
        
        // Wait for video to be ready
        return new Promise((resolve) => {
            elements.video.onloadedmetadata = () => {
                elements.video.play();
                
                // Update canvas dimensions to match video
                const videoWidth = elements.video.videoWidth;
                const videoHeight = elements.video.videoHeight;
                elements.canvas.width = videoWidth;
                elements.canvas.height = videoHeight;
                state.camera.width = videoWidth;
                state.camera.height = videoHeight;
                
                // Enable tracking button
                elements.startCameraBtn.disabled = true;
                elements.startTrackingBtn.disabled = false;
                elements.resetCounterBtn.disabled = false;
                
                // Load PoseNet model
                loadPoseNetModel();
                resolve();
            };
        });
    } catch (error) {
        console.error('Error accessing camera:', error);
        elements.loadingOverlay.style.display = 'none';
        elements.permissionModal.style.display = 'flex';
    }
}

/**
 * Load PoseNet ML model
 */
async function loadPoseNetModel() {
    try {
        // Load a PoseNet model with architecture settings
        state.tracking.model = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: { width: state.camera.width, height: state.camera.height },
            multiplier: 0.75,
            quantBytes: 2
        });
        
        elements.loadingOverlay.style.display = 'none';
        
        // Start a single detection to initialize the system
        detectPose();
        
        // Voice guidance for setup completion if enabled
        if (state.ui.voiceGuide) {
            speakText("Camera and AI model ready. Click start tracking to begin counting sit-ups.");
        }
    } catch (error) {
        console.error('Error loading PoseNet model:', error);
        elements.loadingOverlay.style.display = 'none';
        alert('Failed to load the AI model. Please refresh and try again.');
    }
}

/**
 * Toggle pose tracking on/off
 */
function toggleTracking() {
    state.tracking.active = !state.tracking.active;
    
    if (state.tracking.active) {
        // Start tracking
        elements.startTrackingBtn.innerHTML = '<i class="fas fa-pause"></i> Pause Tracking';
        elements.repStatus.textContent = 'Get in position';
        trackPoses();
        
        if (state.ui.voiceGuide) {
            speakText("Tracking started. Get in position for sit-ups.");
        }
    } else {
        // Pause tracking
        elements.startTrackingBtn.innerHTML = '<i class="fas fa-play"></i> Resume Tracking';
        elements.repStatus.textContent = 'Paused';
        
        if (state.ui.voiceGuide) {
            speakText("Tracking paused.");
        }
    }
}

/**
 * Reset the sit-up counter and state
 */
function resetCounter() {
    state.situp.count = 0;
    state.situp.inUpPosition = false;
    state.situp.inDownPosition = true;
    state.situp.tempCount = 0;
    state.situp.previousAngles = [];
    elements.situpCount.textContent = '0';
    elements.repStatus.textContent = 'Ready';
    elements.progressBar.style.width = '0%';
    
    if (state.ui.voiceGuide) {
        speakText("Counter reset. Ready for new session.");
    }
}

/**
 * Main pose tracking loop
 */
async function trackPoses() {
    if (state.tracking.active && state.camera.active) {
        await detectPose();
        requestAnimationFrame(trackPoses);
    }
}

/**
 * Detect human pose in current video frame
 */
async function detectPose() {
    try {
        if (!state.tracking.model || !elements.video.readyState) return;
        
        // Get pose data with keypoint detection
        const pose = await state.tracking.model.estimateSinglePose(
            elements.video, 
            {
                flipHorizontal: state.ui.mirrorMode
            }
        );
        
        // Only process poses with sufficient confidence
        if (pose.score >= state.tracking.confidence) {
            state.tracking.lastPose = pose;
            
            // Process the detected pose for sit-up counting
            if (state.tracking.active) {
                processDetectedPose(pose);
            }
        }
        
        // Draw the pose on canvas
        drawPose(pose);
    } catch (error) {
        console.error('Error detecting pose:', error);
    }
}

/**
 * Process detected pose keypoints to count sit-ups
 */
function processDetectedPose(pose) {
    const keypoints = pose.keypoints;
    
    // Get key body points
    const leftShoulder = getKeypoint(keypoints, 'leftShoulder');
    const rightShoulder = getKeypoint(keypoints, 'rightShoulder');
    const leftHip = getKeypoint(keypoints, 'leftHip');
    const rightHip = getKeypoint(keypoints, 'rightHip');
    const leftKnee = getKeypoint(keypoints, 'leftKnee');
    const rightKnee = getKeypoint(keypoints, 'rightKnee');
    
    // Skip processing if any essential points are missing
    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip || !leftKnee || !rightKnee) {
        elements.repStatus.textContent = 'Position not clear';
        return;
    }
    
    // Use the side with higher confidence (left or right)
    const shoulder = leftShoulder.score > rightShoulder.score ? leftShoulder : rightShoulder;
    const hip = leftHip.score > rightHip.score ? leftHip : rightHip;
    const knee = leftKnee.score > rightKnee.score ? leftKnee : rightKnee;
    
    // Calculate angle between torso and legs
    const angle = calculateAngle(
        shoulder.position,
        hip.position,
        knee.position
    );
    
    // Smooth angle value using a rolling average
    state.situp.previousAngles.push(angle);
    if (state.situp.previousAngles.length > 5) {
        state.situp.previousAngles.shift();
    }
    
    // Calculate smoothed angle
    const smoothedAngle = state.situp.previousAngles.reduce((a, b) => a + b, 0) / state.situp.previousAngles.length;
    state.situp.angleTorsoLeg = smoothedAngle;
    
    // Update progress bar visualization
    updateSitupProgress(smoothedAngle);
    
    // Detect sit-up state transitions
    detectSitupState(smoothedAngle);
}

/**
 * Helper to get a specific keypoint from the array
 */
function getKeypoint(keypoints, name) {
    const point = keypoints.find(kp => kp.part === name);
    return point && point.score > state.tracking.confidence ? point : null;
}

/**
 * Calculate angle between three points in degrees
 */
function calculateAngle(pointA, pointB, pointC) {
    const angleRadians = Math.atan2(pointC.y - pointB.y, pointC.x - pointB.x) - 
                         Math.atan2(pointA.y - pointB.y, pointA.x - pointB.x);
    let angleDegrees = Math.abs(angleRadians * 180.0 / Math.PI);
    
    // Ensure angle is in the range [0, 180]
    if (angleDegrees > 180.0) {
        angleDegrees = 360.0 - angleDegrees;
    }
    
    return angleDegrees;
}

/**
 * Update the sit-up progress visualization
 */
function updateSitupProgress(angle) {
    // Calculate progress percentage for the progress bar
    // Down position is ~170-180 degrees, up position is ~90-100 degrees
    // Map this range to 0-100% for the progress bar
    let progress = 0;
    
    if (angle >= 170) {
        // Full down position
        progress = 0;
    } else if (angle <= 100) {
        // Full up position
        progress = 100;
    } else {
        // Mapping the angle to progress percentage: (180 - angle) / (180 - 90) * 100
        progress = (180 - angle) / 0.9;
    }
    
    // Update the UI progress bar
    elements.progressBar.style.width = `${progress}%`;
    
    // Change progress bar color based on progress
    if (progress > 90) {
        elements.progressBar.style.backgroundColor = '#4CAF50'; // Green for complete
    } else if (progress > 50) {
        elements.progressBar.style.backgroundColor = '#FFC107'; // Yellow for halfway
    } else {
        elements.progressBar.style.backgroundColor = '#2196F3'; // Blue for starting
    }
}

/**
 * Detect the current state of the sit-up and count repetitions
 */
function detectSitupState(angle) {
    // Constants for angle thresholds
    const DOWN_ANGLE = 160; // Almost flat
    const UP_ANGLE = 100 + state.situp.threshold; // Sitting up, adjusted by sensitivity
    
    // Check transitions between down and up positions
    if (!state.situp.inUpPosition && angle <= UP_ANGLE) {
        // Transition to up position
        state.situp.inUpPosition = true;
        state.situp.inDownPosition = false;
        elements.repStatus.textContent = 'Up position';
        
        // If we were previously in down position, this completes a rep
        if (state.situp.tempCount === 0) {
            // Start counting as halfway through the rep
            state.situp.tempCount = 0.5;
        }
    }
    else if (!state.situp.inDownPosition && angle >= DOWN_ANGLE) {
        // Transition to down position
        state.situp.inDownPosition = true;
        state.situp.inUpPosition = false;
        elements.repStatus.textContent = 'Down position';
        
        // If we were previously in up position, this completes a rep
        if (state.situp.tempCount === 0.5) {
            // Complete the rep
            state.situp.count++;
            state.situp.tempCount = 0;
            updateSitupCount();
            
            // Play sound if enabled
            if (state.ui.audioFeedback) {
                elements.repCompleteSound.play();
            }
        }
    }
    else if (angle > UP_ANGLE && angle < DOWN_ANGLE) {
        // In transition between positions
        elements.repStatus.textContent = 'Moving';
    }
}

/**
 * Update the sit-up counter and statistics
 */
function updateSitupCount() {
    // Update counter in the UI
    elements.situpCount.textContent = state.situp.count;
    
    // Check for milestone achievements
    if (state.situp.count > 0 && state.situp.count % 10 === 0) {
        if (state.ui.audioFeedback) {
            elements.milestoneSound.play();
        }
        
        if (state.ui.voiceGuide) {
            speakText(`Great job! ${state.situp.count} sit-ups completed.`);
        }
    }
    
    // Update statistics
    updateStats();
}

/**
 * Draw the detected pose on the canvas
 */
function drawPose(pose) {
    const ctx = elements.canvas.getContext('2d');
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    
    if (!pose || pose.score < state.tracking.confidence) return;
    
    // Draw skeleton if enabled
    if (state.ui.showSkeleton) {
        drawSkeleton(ctx, pose);
    }
    
    // Draw keypoints
    drawKeypoints(ctx, pose);
    
    // Draw angles if enabled
    if (state.ui.showAngles && state.tracking.active) {
        drawAngles(ctx, pose);
    }
}

/**
 * Draw the skeleton connections between keypoints
 */
function drawSkeleton(ctx, pose) {
    const keypoints = pose.keypoints;
    
    // Set line styling
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    
    // Draw each connection in the skeleton
    for (const [partA, partB] of POSE_CONNECTIONS) {
        const pointA = keypoints.find(kp => kp.part === partA);
        const pointB = keypoints.find(kp => kp.part === partB);
        
        // Only draw if both points are detected with sufficient confidence
        if (pointA && pointB && 
            pointA.score > state.tracking.confidence && 
            pointB.score > state.tracking.confidence) {
            
            ctx.beginPath();
            ctx.moveTo(pointA.position.x, pointA.position.y);
            ctx.lineTo(pointB.position.x, pointB.position.y);
            ctx.stroke();
        }
    }
}

/**
 * Draw the pose keypoints
 */
function drawKeypoints(ctx, pose) {
    pose.keypoints.forEach(keypoint => {
        // Only draw keypoints with sufficient confidence
        if (keypoint.score > state.tracking.confidence) {
            // Larger circles for main tracking points
            const isMainPoint = ['leftShoulder', 'rightShoulder', 'leftHip', 
                                'rightHip', 'leftKnee', 'rightKnee'].includes(keypoint.part);
            
            const radius = isMainPoint ? 8 : 4;
            
            // Different colors for different body parts
            if (keypoint.part.includes('Shoulder')) {
                ctx.fillStyle = '#FF0000'; // Red for shoulders
            } else if (keypoint.part.includes('Hip')) {
                ctx.fillStyle = '#00FF00'; // Green for hips
            } else if (keypoint.part.includes('Knee')) {
                ctx.fillStyle = '#0000FF'; // Blue for knees
            } else {
                ctx.fillStyle = '#FFFF00'; // Yellow for other points
            }
            
            ctx.beginPath();
            ctx.arc(keypoint.position.x, keypoint.position.y, radius, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}

/**
 * Draw the angles relevant to sit-up detection
 */
function drawAngles(ctx, pose) {
    const keypoints = pose.keypoints;
    
    // Get key points for angle calculations
    const leftShoulder = keypoints.find(kp => kp.part === 'leftShoulder');
    const rightShoulder = keypoints.find(kp => kp.part === 'rightShoulder');
    const leftHip = keypoints.find(kp => kp.part === 'leftHip');
    const rightHip = keypoints.find(kp => kp.part === 'rightHip');
    const leftKnee = keypoints.find(kp => kp.part === 'leftKnee');
    const rightKnee = keypoints.find(kp => kp.part === 'rightKnee');
    
    // Draw torso-leg angle (primary angle for sit-up detection)
    // Use the side with higher confidence
    const shoulder = leftShoulder && rightShoulder ? 
                    (leftShoulder.score > rightShoulder.score ? leftShoulder : rightShoulder) : 
                    (leftShoulder || rightShoulder);
    
    const hip = leftHip && rightHip ? 
                (leftHip.score > rightHip.score ? leftHip : rightHip) : 
                (leftHip || rightHip);
    
    const knee = leftKnee && rightKnee ? 
                (leftKnee.score > rightKnee.score ? leftKnee : rightKnee) : 
                (leftKnee || rightKnee);
    
    // Draw angle if all necessary points are available
    if (shoulder && hip && knee && 
        shoulder.score > state.tracking.confidence && 
        hip.score > state.tracking.confidence && 
        knee.score > state.tracking.confidence) {
        
        // Calculate angle
        const angle = Math.round(calculateAngle(
            shoulder.position,
            hip.position,
            knee.position
        ));
        
        // Draw arc and text to visualize angle
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        
        // Draw angle arc
        const radius = 30;
        const startAngle = Math.atan2(shoulder.position.y - hip.position.y, 
                                     shoulder.position.x - hip.position.x);
        const endAngle = Math.atan2(knee.position.y - hip.position.y, 
                                   knee.position.x - hip.position.x);
        
        ctx.beginPath();
        ctx.arc(hip.position.x, hip.position.y, radius, startAngle, endAngle);
        ctx.stroke();
        
        // Draw angle text with background for readability
        const textX = hip.position.x + radius * 1.5 * Math.cos((startAngle + endAngle) / 2);
        const textY = hip.position.y + radius * 1.5 * Math.sin((startAngle + endAngle) / 2);
        
        // Text background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(textX - 25, textY - 15, 50, 20);
        
        // Angle text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '16px Roboto, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${angle}°`, textX, textY);
    }
}

/**
 * Update settings when confidence slider changes
 */
function updateConfidence() {
    // Update confidence value from the range input
    state.tracking.confidence = parseFloat(elements.detectionConfidence.value);
    elements.confidenceValue.textContent = state.tracking.confidence;
    saveSettings();
}

/**
 * Update settings when threshold slider changes
 */
function updateThreshold() {
    // Update angle threshold value from the range input
    state.situp.threshold = parseInt(elements.repThreshold.value);
    elements.thresholdValue.textContent = `${state.situp.threshold}°`;
    saveSettings();
}

/**
 * Update boolean settings (checkboxes)
 */
function updateSettingsBool(event) {
    const setting = event.target.id;
    
    switch(setting) {
        case 'show-skeleton':
            state.ui.showSkeleton = elements.showSkeleton.checked;
            break;
        case 'show-angles':
            state.ui.showAngles = elements.showAngles.checked;
            break;
        case 'mirror-mode':
            state.ui.mirrorMode = elements.mirrorMode.checked;
            break;
        case 'audio-feedback':
            state.ui.audioFeedback = elements.audioFeedback.checked;
            break;
        case 'voice-guide':
            state.ui.voiceGuide = elements.voiceGuide.checked;
            if (state.ui.voiceGuide && state.tracking.active) {
                speakText("Voice guidance activated.");
            }
            break;
    }
    
    saveSettings();
}

/**
 * Save settings to local storage
 */
function saveSettings() {
    const settings = {
        confidence: state.tracking.confidence,
        threshold: state.situp.threshold,
        showSkeleton: state.ui.showSkeleton,
        showAngles: state.ui.showAngles,
        mirrorMode: state.ui.mirrorMode,
        audioFeedback: state.ui.audioFeedback,
        voiceGuide: state.ui.voiceGuide
    };
    
    localStorage.setItem('fitTrackAI_settings', JSON.stringify(settings));
}

/**
 * Load settings from local storage
 */
function loadSettings() {
    const savedSettings = localStorage.getItem('fitTrackAI_settings');
    
    if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        
        // Update state
        state.tracking.confidence = settings.confidence || 0.5;
        state.situp.threshold = settings.threshold || 25;
        state.ui.showSkeleton = settings.showSkeleton !== undefined ? settings.showSkeleton : true;
        state.ui.showAngles = settings.showAngles !== undefined ? settings.showAngles : true;
        state.ui.mirrorMode = settings.mirrorMode !== undefined ? settings.mirrorMode : true;
        state.ui.audioFeedback = settings.audioFeedback !== undefined ? settings.audioFeedback : true;
        state.ui.voiceGuide = settings.voiceGuide !== undefined ? settings.voiceGuide : false;
        
        // Update UI controls
        elements.detectionConfidence.value = state.tracking.confidence;
        elements.confidenceValue.textContent = state.tracking.confidence;
        elements.repThreshold.value = state.situp.threshold;
        elements.thresholdValue.textContent = `${state.situp.threshold}°`;
        elements.showSkeleton.checked = state.ui.showSkeleton;
        elements.showAngles.checked = state.ui.showAngles;
        elements.mirrorMode.checked = state.ui.mirrorMode;
        elements.audioFeedback.checked = state.ui.audioFeedback;
        elements.voiceGuide.checked = state.ui.voiceGuide;
    }
}

/**
 * Update statistics
 */
function updateStats() {
    // Get today's date for stats tracking
    const today = new Date().toLocaleDateString();
    
    // Load existing stats
    let stats = loadStatsData();
    
    // If this is a new day, shift the weekly data
    if (!stats.lastUpdate || stats.lastUpdate !== today) {
        // Shift weekly data left, discarding oldest day
        stats.weekly.shift();
        stats.weekly.push(0);
        stats.today = 0;
    }
    
    // Update stats with new sit-up count
    stats.today = state.situp.count;
    stats.weekly[6] = state.situp.count; // Today is the last element
    
    // Update best if current count is higher
    if (state.situp.count > stats.best) {
        stats.best = state.situp.count;
    }
    
    // Calculate weekly total
    stats.weeklyTotal = stats.weekly.reduce((sum, count) => sum + count, 0);
    
    // Update last update date
    stats.lastUpdate = today;
    
    // Update state
    state.stats.today = stats.today;
    state.stats./**
 * FitTrackAI Sit-up Counter
 * Advanced pose detection and exercise counting application
 */

// Global variables for application state
const state = {
    // Core application state
    camera: {
        active: false,
        stream: null,
        width: 640,
        height: 480
    },
    tracking: {
        active: false,
        model: null,
        lastPose: null,
        confidence: 0.5
    },
    situp: {
        count: 0,
        inUpPosition: false,
        inDownPosition: true,
        angleTorsoLeg: 0,
        threshold: 25, // Default threshold angle in degrees
        tempCount: 0,
        previousAngles: [] // Store previous angles for smoothing
    },
    ui: {
        activeSection: 'workout',
        showSkeleton: true,
        showAngles: true,
        mirrorMode: true,
        audioFeedback: true,
        voiceGuide: false
    },
    stats: {
        today: 0,
        weekly: [0, 0, 0, 0, 0, 0, 0], // Last 7 days
        best: 0,
        chart: null
    }
};

// DOM Elements
const elements = {
    // Main sections
    sections: {
        workout: document.getElementById('workout'),
        stats: document.getElementById('stats'),
        settings: document.getElementById('settings')
    },
    // Nav buttons
    navButtons: document.querySelectorAll('.nav-btn'),
    
    // Camera & tracking
    video: document.getElementById('video'),
    canvas: document.getElementById('output'),
    startCameraBtn: document.getElementById('start-camera'),
    startTrackingBtn: document.getElementById('start-tracking'),
    resetCounterBtn: document.getElementById('reset-counter'),
    
    // Counters & indicators
    situpCount: document.getElementById('sit-up-count'),
    repIndicator: document.getElementById('rep-indicator'),
    repStatus: document.querySelector('.rep-status'),
    progressBar: document.querySelector('.progress-bar'),
    
    // Stats elements
    todayCount: document.getElementById('today-count'),
    weeklyCount: document.getElementById('weekly-count'),
    bestCount: document.getElementById('best-count'),
    weeklyChart: document.getElementById('weekly-chart'),
    
    // Settings elements
    detectionConfidence: document.getElementById('detection-confidence'),
    confidenceValue: document.getElementById('confidence-value'),
    repThreshold: document.getElementById('rep-threshold'),
    thresholdValue: document.getElementById('threshold-value'),
    showSkeleton: document.getElementById('show-skeleton'),
    showAngles: document.getElementById('show-angles'),
    mirrorMode: document.getElementById('mirror-mode'),
    audioFeedback: document.getElementById('audio-feedback'),
    voiceGuide: document.getElementById('voice-guide'),
    
    // Modal and overlay
    permissionModal: document.getElementById('permission-modal'),
    grantPermissionBtn: document.getElementById('grant-permission'),
    closeModalBtn: document.querySelector('.close-modal'),
    loadingOverlay: document.getElementById('loading-overlay'),
    
    // Audio
    repCompleteSound: document.getElementById('rep-complete-sound'),
    milestoneSound: document.getElementById('milestone-sound')
};

// Drawing constants
const POSE_CONNECTIONS = [
    ['leftShoulder', 'rightShoulder'], 
    ['leftShoulder', 'leftHip'], 
    ['rightShoulder', 'rightHip'], 
    ['leftHip', 'rightHip'],
    ['leftShoulder', 'leftElbow'], 
    ['leftElbow', 'leftWrist'],
    ['rightShoulder', 'rightElbow'], 
    ['rightElbow', 'rightWrist'],
    ['leftHip', 'leftKnee'], 
    ['leftKnee', 'leftAnkle'],
    ['rightHip', 'rightKnee'], 
    ['rightKnee', 'rightAnkle']
];

// Speech synthesis for voice guidance
const speechSynthesis = window.speechSynthesis;

/**
 * Application Initialization
 */
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadSettings();
    loadStats();
    initStatsChart();
});

/**
 * Initialize all event listeners
 */
function initEventListeners() {
    // Navigation
    elements.navButtons.forEach(button => {
        button.addEventListener('click', () => switchSection(button.dataset.section));
    });
    
    // Camera controls
    elements.startCameraBtn.addEventListener('click', setupCamera);
    elements.startTrackingBtn.addEventListener('click', toggleTracking);
    elements.resetCounterBtn.addEventListener('click', resetCounter);
    
    // Modal interactions
    elements.grantPermissionBtn.addEventListener('click', setupCamera);
    elements.closeModalBtn.addEventListener('click', () => {
        elements.permissionModal.style.display = 'none';
    });
    
    // Settings changes
    elements.detectionConfidence.addEventListener('input', updateConfidence);
    elements.repThreshold.addEventListener('input', updateThreshold);
    elements.showSkeleton.addEventListener('change', updateSettingsBool);
    elements.showAngles.addEventListener('change', updateSettingsBool);
    elements.mirrorMode.addEventListener('change', updateSettingsBool);
    elements.audioFeedback.addEventListener('change', updateSettingsBool);
    elements.voiceGuide.addEventListener('change', updateSettingsBool);
    
    // Handle clicks outside the modal to close it
    window.addEventListener('click', (event) => {
        if (event.target === elements.permissionModal) {
            elements.permissionModal.style.display = 'none';
        }
    });
}

/**
 * Switch between app sections (workout, stats, settings)
 */
function switchSection(sectionId) {
    // Hide all sections
    Object.values(elements.sections).forEach(section => {
        section.classList.remove('active-section');
    });
    
    // Show selected section
    elements.sections[sectionId].classList.add('active-section');
    
    // Update nav button state
    elements.navButtons.forEach(button => {
        if (button.dataset.section === sectionId) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
    
    // Update current section in state
    state.ui.activeSection = sectionId;
    
    // Refresh charts if switching to stats
    if (sectionId === 'stats' && state.stats.chart) {
        state.stats.chart.update();
    }
}

/**
 * Camera Setup & Permissions
 */
async function setupCamera() {
    elements.permissionModal.style.display = 'none';
    elements.loadingOverlay.style.display = 'flex';
    
    try {
        // Request camera access with constraints
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: 'user',
                width: { ideal: state.camera.width },
                height: { ideal: state.camera.height }
            }
        });
        
        elements.video.srcObject = stream;
        state.camera.stream = stream;
        state.camera.active = true;
        
        // Wait for video to be ready
        return new Promise((resolve) => {
            elements.video.onloadedmetadata = () => {
                elements.video.play();
                
                // Update canvas dimensions to match video
                const videoWidth = elements.video.videoWidth;
                const videoHeight = elements.video.videoHeight;
                elements.canvas.width = videoWidth;
                elements.canvas.height = videoHeight;
                state.camera.width = videoWidth;
                state.camera.height = videoHeight;
                
                // Enable tracking button
                elements.startCameraBtn.disabled = true;
                elements.startTrackingBtn.disabled = false;
                elements.resetCounterBtn.disabled = false;
                
                // Load PoseNet model
                loadPoseNetModel();
                resolve();
            };
        });
    } catch (error) {
        console.error('Error accessing camera:', error);
        elements.loadingOverlay.style.display = 'none';
        elements.permissionModal.style.display = 'flex';
    }
}

/**
 * Load PoseNet ML model
 */
async function loadPoseNetModel() {
    try {
        // Load a PoseNet model with architecture settings
        state.tracking.model = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: { width: state.camera.width, height: state.camera.height },
            multiplier: 0.75,
            quantBytes: 2
        });
        
        elements.loadingOverlay.style.display = 'none';
        
        // Start a single detection to initialize the system
        detectPose();
        
        // Voice guidance for setup completion if enabled
        if (state.ui.voiceGuide) {
            speakText("Camera and AI model ready. Click start tracking to begin counting sit-ups.");
        }
    } catch (error) {
        console.error('Error loading PoseNet model:', error);
        elements.loadingOverlay.style.display = 'none';
        alert('Failed to load the AI model. Please refresh and try again.');
    }
}

/**
 * Toggle pose tracking on/off
 */
function toggleTracking() {
    state.tracking.active = !state.tracking.active;
    
    if (state.tracking.active) {
        // Start tracking
        elements.startTrackingBtn.innerHTML = '<i class="fas fa-pause"></i> Pause Tracking';
        elements.repStatus.textContent = 'Get in position';
        trackPoses();
        
        if (state.ui.voiceGuide) {
            speakText("Tracking started. Get in position for sit-ups.");
        }
    } else {
        // Pause tracking
        elements.startTrackingBtn.innerHTML = '<i class="fas fa-play"></i> Resume Tracking';
        elements.repStatus.textContent = 'Paused';
        
        if (state.ui.voiceGuide) {
            speakText("Tracking paused.");
        }
    }
}

/**
 * Reset the sit-up counter and state
 */
function resetCounter() {
    state.situp.count = 0;
    state.situp.inUpPosition = false;
    state.situp.inDownPosition = true;
    state.situp.tempCount = 0;
    state.situp.previousAngles = [];
    elements.situpCount.textContent = '0';
    elements.repStatus.textContent = 'Ready';
    elements.progressBar.style.width = '0%';
    
    if (state.ui.voiceGuide) {
        speakText("Counter reset. Ready for new session.");
    }
}

/**
 * Main pose tracking loop
 */
async function trackPoses() {
    if (state.tracking.active && state.camera.active) {
        await detectPose();
        requestAnimationFrame(trackPoses);
    }
}

/**
 * Detect human pose in current video frame
 */
async function detectPose() {
    try {
        if (!state.tracking.model || !elements.video.readyState) return;
        
        // Get pose data with keypoint detection
        const pose = await state.tracking.model.estimateSinglePose(
            elements.video, 
            {
                flipHorizontal: state.ui.mirrorMode
            }
        );
        
        // Only process poses with sufficient confidence
        if (pose.score >= state.tracking.confidence) {
            state.tracking.lastPose = pose;
            
            // Process the detected pose for sit-up counting
            if (state.tracking.active) {
                processDetectedPose(pose);
            }
        }
        
        // Draw the pose on canvas
        drawPose(pose);
    } catch (error) {
        console.error('Error detecting pose:', error);
    }
}

/**
 * Process detected pose keypoints to count sit-ups
 */
function processDetectedPose(pose) {
    const keypoints = pose.keypoints;
    
    // Get key body points
    const leftShoulder = getKeypoint(keypoints, 'leftShoulder');
    const rightShoulder = getKeypoint(keypoints, 'rightShoulder');
    const leftHip = getKeypoint(keypoints, 'leftHip');
    const rightHip = getKeypoint(keypoints, 'rightHip');
    const leftKnee = getKeypoint(keypoints, 'leftKnee');
    const rightKnee = getKeypoint(keypoints, 'rightKnee');
    
    // Skip processing if any essential points are missing
    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip || !leftKnee || !rightKnee) {
        elements.repStatus.textContent = 'Position not clear';
        return;
    }
    
    // Use the side with higher confidence (left or right)
    const shoulder = leftShoulder.score > rightShoulder.score ? leftShoulder : rightShoulder;
    const hip = leftHip.score > rightHip.score ? leftHip : rightHip;
    const knee = leftKnee.score > rightKnee.score ? leftKnee : rightKnee;
    
    // Calculate angle between torso and legs
    const angle = calculateAngle(
        shoulder.position,
        hip.position,
        knee.position
    );
    
    // Smooth angle value using a rolling average
    state.situp.previousAngles.push(angle);
    if (state.situp.previousAngles.length > 5) {
        state.situp.previousAngles.shift();
    }
    
    // Calculate smoothed angle
    const smoothedAngle = state.situp.previousAngles.reduce((a, b) => a + b, 0) / state.situp.previousAngles.length;
    state.situp.angleTorsoLeg = smoothedAngle;
    
    // Update progress bar visualization
    updateSitupProgress(smoothedAngle);
    
    // Detect sit-up state transitions
    detectSitupState(smoothedAngle);
}

/**
 * Helper to get a specific keypoint from the array
 */
function getKeypoint(keypoints, name) {
    const point = keypoints.find(kp => kp.part === name);
    return point && point.score > state.tracking.confidence ? point : null;
}

/**
 * Calculate angle between three points in degrees
 */
function calculateAngle(pointA, pointB, pointC) {
    const angleRadians = Math.atan2(pointC.y - pointB.y, pointC.x - pointB.x) - 
                         Math.atan2(pointA.y - pointB.y, pointA.x - pointB.x);
    let angleDegrees = Math.abs(angleRadians * 180.0 / Math.PI);
    
    // Ensure angle is in the range [0, 180]
    if (angleDegrees > 180.0) {
        angleDegrees = 360.0 - angleDegrees;
    }
    
    return angleDegrees;
}

/**
 * Update the sit-up progress visualization
 */
function updateSitupProgress(angle) {
    // Calculate progress percentage for the progress bar
    // Down position is ~170-180 degrees, up position is ~90-100 degrees
    // Map this range to 0-100% for the progress bar
    let progress = 0;
    
    if (angle >= 170) {
        // Full down position
        progress = 0;
    } else if (angle <= 100) {
        // Full up position
        progress = 100;
    } else {
        // Mapping the angle to progress percentage: (180 - angle) / (180 - 90) * 100
        progress = (180 - angle) / 0.9;
    }
    
    // Update the UI progress bar
    elements.progressBar.style.width = `${progress}%`;
    
    // Change progress bar color based on progress
    if (progress > 90) {
        elements.progressBar.style.backgroundColor = '#4CAF50'; // Green for complete
    } else if (progress > 50) {
        elements.progressBar.style.backgroundColor = '#FFC107'; // Yellow for halfway
    } else {
        elements.progressBar.style.backgroundColor = '#2196F3'; // Blue for starting
    }
}

/**
 * Detect the current state of the sit-up and count repetitions
 */
function detectSitupState(angle) {
    // Constants for angle thresholds
    const DOWN_ANGLE = 160; // Almost flat
    const UP_ANGLE = 100 + state.situp.threshold; // Sitting up, adjusted by sensitivity
    
    // Check transitions between down and up positions
    if (!state.situp.inUpPosition && angle <= UP_ANGLE) {
        // Transition to up position
        state.situp.inUpPosition = true;
        state.situp.inDownPosition = false;
        elements.repStatus.textContent = 'Up position';
        
        // If we were previously in down position, this completes a rep
        if (state.situp.tempCount === 0) {
            // Start counting as halfway through the rep
            state.situp.tempCount = 0.5;
        }
    }
    else if (!state.situp.inDownPosition && angle >= DOWN_ANGLE) {
        // Transition to down position
        state.situp.inDownPosition = true;
        state.situp.inUpPosition = false;
        elements.repStatus.textContent = 'Down position';
        
        // If we were previously in up position, this completes a rep
        if (state.situp.tempCount === 0.5) {
            // Complete the rep
            state.situp.count++;
            state.situp.tempCount = 0;
            updateSitupCount();
            
            // Play sound if enabled
            if (state.ui.audioFeedback) {
                elements.repCompleteSound.play();
            }
        }
    }
    else if (angle > UP_ANGLE && angle < DOWN_ANGLE) {
        // In transition between positions
        elements.repStatus.textContent = 'Moving';
    }
}

/**
 * Update the sit-up counter and statistics
 */
function updateSitupCount() {
    // Update counter in the UI
    elements.situpCount.textContent = state.situp.count;
    
    // Check for milestone achievements
    if (state.situp.count > 0 && state.situp.count % 10 === 0) {
        if (state.ui.audioFeedback) {
            elements.milestoneSound.play();
        }
        
        if (state.ui.voiceGuide) {
            speakText(`Great job! ${state.situp.count} sit-ups completed.`);
        }
    }
    
    // Update statistics
    updateStats();
}

/**
 * Draw the detected pose on the canvas
 */
function drawPose(pose) {
    const ctx = elements.canvas.getContext('2d');
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    
    if (!pose || pose.score < state.tracking.confidence) return;
    
    // Draw skeleton if enabled
    if (state.ui.showSkeleton) {
        drawSkeleton(ctx, pose);
    }
    
    // Draw keypoints
    drawKeypoints(ctx, pose);
    
    // Draw angles if enabled
    if (state.ui.showAngles && state.tracking.active) {
        drawAngles(ctx, pose);
    }
}

/**
 * Draw the skeleton connections between keypoints
 */
function drawSkeleton(ctx, pose) {
    const keypoints = pose.keypoints;
    
    // Set line styling
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    
    // Draw each connection in the skeleton
    for (const [partA, partB] of POSE_CONNECTIONS) {
        const pointA = keypoints.find(kp => kp.part === partA);
        const pointB = keypoints.find(kp => kp.part === partB);
        
        // Only draw if both points are detected with sufficient confidence
        if (pointA && pointB && 
            pointA.score > state.tracking.confidence && 
            pointB.score > state.tracking.confidence) {
            
            ctx.beginPath();
            ctx.moveTo(pointA.position.x, pointA.position.y);
            ctx.lineTo(pointB.position.x, pointB.position.y);
            ctx.stroke();
        }
    }
}

/**
 * Draw the pose keypoints
 */
function drawKeypoints(ctx, pose) {
    pose.keypoints.forEach(keypoint => {
        // Only draw keypoints with sufficient confidence
        if (keypoint.score > state.tracking.confidence) {
            // Larger circles for main tracking points
            const isMainPoint = ['leftShoulder', 'rightShoulder', 'leftHip', 
                                'rightHip', 'leftKnee', 'rightKnee'].includes(keypoint.part);
            
            const radius = isMainPoint ? 8 : 4;
            
            // Different colors for different body parts
            if (keypoint.part.includes('Shoulder')) {
                ctx.fillStyle = '#FF0000'; // Red for shoulders
            } else if (keypoint.part.includes('Hip')) {
                ctx.fillStyle = '#00FF00'; // Green for hips
            } else if (keypoint.part.includes('Knee')) {
                ctx.fillStyle = '#0000FF'; // Blue for knees
            } else {
                ctx.fillStyle = '#FFFF00'; // Yellow for other points
            }
            
            ctx.beginPath();
            ctx.arc(keypoint.position.x, keypoint.position.y, radius, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}

/**
 * Draw the angles relevant to sit-up detection
 */
function drawAngles(ctx, pose) {
    const keypoints = pose.keypoints;
    
    // Get key points for angle calculations
    const leftShoulder = keypoints.find(kp => kp.part === 'leftShoulder');
    const rightShoulder = keypoints.find(kp => kp.part === 'rightShoulder');
    const leftHip = keypoints.find(kp => kp.part === 'leftHip');
    const rightHip = keypoints.find(kp => kp.part === 'rightHip');
    const leftKnee = keypoints.find(kp => kp.part === 'leftKnee');
    const rightKnee = keypoints.find(kp => kp.part === 'rightKnee');
    
    // Draw torso-leg angle (primary angle for sit-up detection)
    // Use the side with higher confidence
    const shoulder = leftShoulder && rightShoulder ? 
                    (leftShoulder.score > rightShoulder.score ? leftShoulder : rightShoulder) : 
                    (leftShoulder || rightShoulder);
    
    const hip = leftHip && rightHip ? 
                (leftHip.score > rightHip.score ? leftHip : rightHip) : 
                (leftHip || rightHip);
    
    const knee = leftKnee && rightKnee ? 
                (leftKnee.score > rightKnee.score ? leftKnee : rightKnee) : 
                (leftKnee || rightKnee);
    
    // Draw angle if all necessary points are available
    if (shoulder && hip && knee && 
        shoulder.score > state.tracking.confidence && 
        hip.score > state.tracking.confidence && 
        knee.score > state.tracking.confidence) {
        
        // Calculate angle
        const angle = Math.round(calculateAngle(
            shoulder.position,
            hip.position,
            knee.position
        ));
        
        // Draw arc and text to visualize angle
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        
        // Draw angle arc
        const radius = 30;
        const startAngle = Math.atan2(shoulder.position.y - hip.position.y, 
                                     shoulder.position.x - hip.position.x);
        const endAngle = Math.atan2(knee.position.y - hip.position.y, 
                                   knee.position.x - hip.position.x);
        
        ctx.beginPath();
        ctx.arc(hip.position.x, hip.position.y, radius, startAngle, endAngle);
        ctx.stroke();
        
        // Draw angle text with background for readability
        const textX = hip.position.x + radius * 1.5 * Math.cos((startAngle + endAngle) / 2);
        const textY = hip.position.y + radius * 1.5 * Math.sin((startAngle + endAngle) / 2);
        
        // Text background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(textX - 25, textY - 15, 50, 20);
        
        // Angle text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '16px Roboto, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${angle}°`, textX, textY);
    }
}

/**
 * Update settings when confidence slider changes
 */
function updateConfidence() {
    // Update confidence value from the range input
    state.tracking.confidence = parseFloat(elements.detectionConfidence.value);
    elements.confidenceValue.textContent = state.tracking.confidence;
    saveSettings();
}

/**
 * Update settings when threshold slider changes
 */
function updateThreshold() {
    // Update angle threshold value from the range input
    state.situp.threshold = parseInt(elements.repThreshold.value);
    elements.thresholdValue.textContent = `${state.situp.threshold}°`;
    saveSettings();
}

/**
 * Update boolean settings (checkboxes)
 */
function updateSettingsBool(event) {
    const setting = event.target.id;
    
    switch(setting) {
        case 'show-skeleton':
            state.ui.showSkeleton = elements.showSkeleton.checked;
            break;
        case 'show-angles':
            state.ui.showAngles = elements.showAngles.checked;
            break;
        case 'mirror-mode':
            state.ui.mirrorMode = elements.mirrorMode.checked;
            break;
        case 'audio-feedback':
            state.ui.audioFeedback = elements.audioFeedback.checked;
            break;
        case 'voice-guide':
            state.ui.voiceGuide = elements.voiceGuide.checked;
            if (state.ui.voiceGuide && state.tracking.active) {
                speakText("Voice guidance activated.");
            }
            break;
    }
    
    saveSettings();
}

/**
 * Save settings to local storage
 */
function saveSettings() {
    const settings = {
        confidence: state.tracking.confidence,
        threshold: state.situp.threshold,
        showSkeleton: state.ui.showSkeleton,
        showAngles: state.ui.showAngles,
        mirrorMode: state.ui.mirrorMode,
        audioFeedback: state.ui.audioFeedback,
        voiceGuide: state.ui.voiceGuide
    };
    
    localStorage.setItem('fitTrackAI_settings', JSON.stringify(settings));
}

/**
 * Load settings from local storage
 */
function loadSettings() {
    const savedSettings = localStorage.getItem('fitTrackAI_settings');
    
    if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        
        // Update state
        state.tracking.confidence = settings.confidence || 0.5;
        state.situp.threshold = settings.threshold || 25;
        state.ui.showSkeleton = settings.showSkeleton !== undefined ? settings.showSkeleton : true;
        state.ui.showAngles = settings.showAngles !== undefined ? settings.showAngles : true;
        state.ui.mirrorMode = settings.mirrorMode !== undefined ? settings.mirrorMode : true;
        state.ui.audioFeedback = settings.audioFeedback !== undefined ? settings.audioFeedback : true;
        state.ui.voiceGuide = settings.voiceGuide !== undefined ? settings.voiceGuide : false;
        
        // Update UI controls
        elements.detectionConfidence.value = state.tracking.confidence;
        elements.confidenceValue.textContent = state.tracking.confidence;
        elements.repThreshold.value = state.situp.threshold;
        elements.thresholdValue.textContent = `${state.situp.threshold}°`;
        elements.showSkeleton.checked = state.ui.showSkeleton;
        elements.showAngles.checked = state.ui.showAngles;
        elements.mirrorMode.checked = state.ui.mirrorMode;
        elements.audioFeedback.checked = state.ui.audioFeedback;
        elements.voiceGuide.checked = state.ui.voiceGuide;
    }
}

/**
 * Update statistics
 */
function updateStats() {
    // Get today's date for stats tracking
    const today = new Date().toLocaleDateString();
    
    // Load existing stats
    let stats = loadStatsData();
    
    // If this is a new day, shift the weekly data
    if (!stats.lastUpdate || stats.lastUpdate !== today) {
        // Shift weekly data left, discarding oldest day
        stats.weekly.shift();
        stats.weekly.push(0);
        stats.today = 0;
    }
    
    // Update stats with new sit-up count
    stats.today = state.situp.count;
    stats.weekly[6] = state.situp.count; // Today is the last element
    
    // Update best if current count is higher
    if (state.situp.count > stats.best) {
        stats.best = state.situp.count;
    }
    
    // Calculate weekly total
    stats.weeklyTotal = stats.weekly.reduce((sum, count) => sum + count, 0);
    
    // Update last update date
    stats.lastUpdate = today;
    
    // Update state
    state.stats.today = stats.today;
    state.stats.weekly = stats.weekly;
    state.stats.best = stats.best;
    
    // Update UI elements
    elements.todayCount.textContent = stats.today;
    elements.weeklyCount.textContent = stats.weeklyTotal;
    elements.bestCount.textContent = stats.best;
    
    // Update chart if it exists
    if (state.stats.chart) {
        updateStatsChart();
    }
    
    // Save stats to local storage
    saveStatsData(stats);
}

/**
 * Load statistics from local storage
 */
function loadStatsData() {
    const savedStats = localStorage.getItem('fitTrackAI_stats');
    
    // Default stats structure
    const defaultStats = {
        today: 0,
        weekly: [0, 0, 0, 0, 0, 0, 0],
        weeklyTotal: 0,
        best: 0,
        lastUpdate: null
    };
    
    if (savedStats) {
        try {
            // Parse saved stats and use them
            return JSON.parse(savedStats);
        } catch (error) {
            console.error('Error parsing saved stats:', error);
            return defaultStats;
        }
    }
    
    return defaultStats;
}

/**
 * Save statistics to local storage
 */
function saveStatsData(stats) {
    localStorage.setItem('fitTrackAI_stats', JSON.stringify(stats));
}

/**
 * Load statistics from local storage and update state and UI
 */
function loadStats() {
    const stats = loadStatsData();
    
    // Update state
    state.stats.today = stats.today;
    state.stats.weekly = stats.weekly;
    state.stats.best = stats.best;
    
    // Update UI
    elements.todayCount.textContent = stats.today;
    elements.weeklyCount.textContent = stats.weekly.reduce((sum, count) => sum + count, 0);
    elements.bestCount.textContent = stats.best;
}

/**
 * Initialize the weekly progress chart
 */
function initStatsChart() {
    if (!elements.weeklyChart) return;
    
    // Get the day labels for the last 7 days
    const dayLabels = getLast7DayLabels();
    
    const chartConfig = {
        type: 'bar',
        data: {
            labels: dayLabels,
            datasets: [{
                label: 'Sit-ups',
                data: state.stats.weekly,
                backgroundColor: 'rgba(33, 150, 243, 0.7)',
                borderColor: 'rgba(33, 150, 243, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            return dayLabels[context[0].dataIndex];
                        },
                        label: function(context) {
                            return context.parsed.y + ' sit-ups';
                        }
                    }
                }
            }
        }
    };
    
    // Create the chart
    state.stats.chart = new Chart(elements.weeklyChart.getContext('2d'), chartConfig);
}

/**
 * Update the weekly progress chart
 */
function updateStatsChart() {
    if (!state.stats.chart) return;
    
    // Update chart data
    state.stats.chart.data.datasets[0].data = state.stats.weekly;
    
    // Update labels for the last 7 days
    state.stats.chart.data.labels = getLast7DayLabels();
    
    // Update chart
    state.stats.chart.update();
}

/**
 * Get labels for the last 7 days (including today)
 */
function getLast7DayLabels() {
    const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const today = new Date();
    const dayLabels = [];
    
    // Generate labels for the past 7 days (including today)
    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(today.getDate() - i);
        dayLabels.push(days[date.getDay()]);
    }
    
    return dayLabels;
}

/**
 * Speak text using speech synthesis
 */
function speakText(text) {
    // Cancel any ongoing speech
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
    }
    
    // Create a new speech utterance
    const utterance = new SpeechSynthesisUtterance(text);
    
    // Use a slightly slower rate for better clarity
    utterance.rate = 0.9;
    utterance.pitch = 1.1;
    
    // Speak the text
    speechSynthesis.speak(utterance);
}

/**
 * Form correction and guidance based on pose analysis
 */
function analyzeFormQuality(pose) {
    if (!pose || !state.tracking.active) return;
    
    const keypoints = pose.keypoints;
    
    // Extract key points for form analysis
    const nose = getKeypoint(keypoints, 'nose');
    const leftShoulder = getKeypoint(keypoints, 'leftShoulder');
    const rightShoulder = getKeypoint(keypoints, 'rightShoulder');
    const leftHip = getKeypoint(keypoints, 'leftHip');
    const rightHip = getKeypoint(keypoints, 'rightHip');
    const leftKnee = getKeypoint(keypoints, 'leftKnee');
    const rightKnee = getKeypoint(keypoints, 'rightKnee');
    
    // Skip if essential points are missing
    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip || !leftKnee || !rightKnee) {
        return;
    }
    
    // Check if head is properly aligned
    if (nose && state.situp.inUpPosition) {
        const headPosition = (nose.position.x > leftShoulder.position.x && 
                                nose.position.x < rightShoulder.position.x);
        
        if (!headPosition && state.ui.voiceGuide) {
            speakText("Keep your head aligned with your torso");
        }
    }
    
    // Check symmetry in sit-up movement (both sides moving equally)
    if (state.situp.inUpPosition) {
        const leftSide = calculateAngle(leftShoulder.position, leftHip.position, leftKnee.position);
        const rightSide = calculateAngle(rightShoulder.position, rightHip.position, rightKnee.position);
        
        // If asymmetry detected
        if (Math.abs(leftSide - rightSide) > 15 && state.ui.voiceGuide) {
            speakText("Try to keep your movement balanced on both sides");
        }
    }
}

/**
 * Clean up resources when the app is closed/refreshed
 */
window.addEventListener('beforeunload', () => {
    // Stop camera stream if active
    if (state.camera.active && state.camera.stream) {
        const tracks = state.camera.stream.getTracks();
        tracks.forEach(track => track.stop());
    }
    
    // Cancel any ongoing speech
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
    }
});

/**
 * Handle visibility changes (tab switching, etc.)
 */
document.addEventListener('visibilitychange', () => {
    // Pause tracking when tab is hidden
    if (document.hidden && state.tracking.active) {
        toggleTracking(); // Pause tracking
    }
});

/**
 * Handle window resize events
 */
window.addEventListener('resize', () => {
    // Update chart if it exists
    if (state.stats.chart) {
        state.stats.chart.resize();
    }
});

/**
 * Detect mobile devices
 */
function isMobileDevice() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

/**
 * Initialize mobile-specific adjustments
 */
function initMobileAdjustments() {
    if (isMobileDevice()) {
        // Adjust camera resolution for better performance on mobile
        state.camera.width = 480;
        state.camera.height = 360;
        
        // Add landscape orientation warning if needed
        window.addEventListener('orientationchange', checkOrientation);
        checkOrientation();
    }
}

/**
 * Check device orientation and show warning if needed
 */
function checkOrientation() {
    const orientationWarning = document.createElement('div');
    orientationWarning.id = 'orientation-warning';
    orientationWarning.innerHTML = `
        <div class="warning-content">
            <i class="fas fa-mobile-alt"></i>
            <p>Please rotate your device to portrait mode for better experience</p>
        </div>
    `;
    orientationWarning.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.9);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        text-align: center;
        font-size: 1.2rem;
    `;
    
    // Check if we're in landscape mode on a mobile device
    if (window.innerWidth > window.innerHeight) {
        document.body.appendChild(orientationWarning);
    } else {
        const existingWarning = document.getElementById('orientation-warning');
        if (existingWarning) {
            existingWarning.remove();
        }
    }
}

// Initialize mobile adjustments
initMobileAdjustments();