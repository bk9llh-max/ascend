// ascend.js - Pure browser-based looksmaxxing analyzer
// No server required - 100% client-side processing

// Initialize Human library for facial analysis
let human = null;
let modelsLoaded = false;

// Configuration for Human library
const humanConfig = {
    backend: 'webgl', // Use WebGL for GPU acceleration
    async: true,
    warmup: 'none',
    cacheSensitivity: 0.9,
    debug: false,
    modelBasePath: 'https://cdn.jsdelivr.net/npm/@vladmandic/human@0.40.6/models/',
    face: {
        enabled: true,
        detector: { 
            enabled: true, 
            maxDetected: 1,
            minConfidence: 0.5,
            model: 'blazeface-back' // Fast and accurate face detection [citation:10]
        },
        mesh: { 
            enabled: true, 
            model: 'facemesh' // 468-point 3D facial landmarks [citation:2]
        },
        iris: { enabled: false },
        description: { enabled: false },
        emotion: { enabled: false },
        age: { enabled: false },
        gender: { enabled: false }
    },
    body: { enabled: false },
    hand: { enabled: false }
};

// Looksmaxxing categories
const LOOKS_CATEGORIES = [
    { min: 98, name: 'CHAD PREMIUM' },
    { min: 95, name: 'CHAD' },
    { min: 92, name: 'HTN+' },
    { min: 89, name: 'HTN' },
    { min: 86, name: 'HTN-' },
    { min: 83, name: 'MTN+' },
    { min: 80, name: 'MTN' },
    { min: 77, name: 'MTN-' },
    { min: 74, name: 'LTN+' },
    { min: 71, name: 'LTN' },
    { min: 68, name: 'LTN-' },
    { min: 65, name: 'Sub5+' },
    { min: 62, name: 'Sub5' },
    { min: 59, name: 'Sub5-' },
    { min: 55, name: 'Sub4' },
    { min: 50, name: 'Sub3' },
    { min: 0, name: 'Sub2' }
];

// Golden ratio reference values
const IDEAL_RATIOS = {
    fwhr: 0.75,        // Face width-to-height ratio
    eyeSpacing: 0.46,   // Eye spacing ratio
    noseWidth: 0.25,    // Nose width to face width
    lipFullness: 0.35,  // Lip height to width
    canthalTilt: 8,     // Degrees (positive tilt)
    gonialAngle: 125    // Degrees (jaw angle)
};

// Initialize Human library
async function initHuman() {
    try {
        human = new Human(humanConfig);
        await human.load();
        modelsLoaded = true;
        console.log('Human models loaded successfully');
        return true;
    } catch (error) {
        console.error('Failed to load Human models:', error);
        return false;
    }
}

// Main analysis function
async function analyzeFace(imageElement) {
    if (!modelsLoaded) {
        await initHuman();
    }
    
    if (!modelsLoaded) {
        throw new Error('Failed to load analysis models');
    }
    
    // Run detection using Human library [citation:10]
    const result = await human.detect(imageElement);
    
    if (!result.face || result.face.length === 0) {
        throw new Error('No face detected. Please ensure good lighting and face is clearly visible.');
    }
    
    const face = result.face[0];
    const landmarks = face.mesh; // 468 3D landmarks [citation:2]
    
    if (!landmarks || landmarks.length < 468) {
        throw new Error('Insufficient facial landmarks detected');
    }
    
    // Calculate all metrics
    const metrics = calculateAllMetrics(landmarks, face);
    
    // Calculate weighted overall score
    const weights = {
        symmetry: 0.20,
        fwhr: 0.15,
        jawline: 0.15,
        browRidge: 0.10,
        chin: 0.10,
        eyeArea: 0.10,
        noseShape: 0.05,
        lipFullness: 0.05,
        facialThirds: 0.05,
        canthalTilt: 0.03,
        midfaceRatio: 0.01,
        gonialAngle: 0.01
    };
    
    const overallScore = Object.keys(weights).reduce((sum, key) => {
        return sum + (metrics[key] || 50) * weights[key];
    }, 0);
    
    // Determine category
    const category = getLooksCategory(overallScore);
    
    // Calculate potential
    const potential = calculatePotential(metrics);
    
    // Generate flaws and strengths
    const { flaws, strengths } = analyzeFlawsStrengths(metrics);
    
    // Generate recommendations
    const recommendations = generateRecommendations(metrics);
    
    return {
        success: true,
        metrics,
        overall: overallScore,
        category,
        potential,
        flaws,
        strengths,
        recommendations
    };
}

function calculateAllMetrics(landmarks, face) {
    // Convert landmarks to pixel coordinates
    const points = landmarks.map(p => ({ x: p[0], y: p[1], z: p[2] || 0 }));
    
    return {
        symmetry: calculateSymmetryScore(points),
        fwhr: calculateFwhr(points),
        jawline: calculateJawlineScore(points),
        browRidge: calculateBrowRidgeScore(points),
        chin: calculateChinScore(points),
        eyeArea: calculateEyeAreaScore(points),
        noseShape: calculateNoseScore(points),
        lipFullness: calculateLipScore(points),
        facialThirds: calculateFacialThirdsScore(points),
        canthalTilt: calculateCanthalTilt(points),
        midfaceRatio: calculateMidfaceRatio(points),
        gonialAngle: calculateGonialAngle(points)
    };
}

function calculateSymmetryScore(points) {
    // Define symmetry point pairs (left/right indices based on MediaPipe 468 landmarks)
    const symmetryPairs = [
        [33, 263], [133, 362], [159, 386], [145, 374],  // Eyes
        [61, 291], [37, 267], [0, 17], [1, 15],          // Mouth and face outline
        [70, 300], [63, 293], [105, 334], [66, 296],     // Brows
        [49, 279], [53, 283], [55, 285], [59, 289]       // Nose and mouth
    ];
    
    let scores = [];
    
    symmetryPairs.forEach(([leftIdx, rightIdx]) => {
        if (leftIdx < points.length && rightIdx < points.length) {
            const left = points[leftIdx];
            const right = points[rightIdx];
            
            // Mirror right point across vertical axis
            const midX = (left.x + right.x) / 2;
            const mirroredRight = { x: 2 * midX - right.x, y: right.y };
            
            // Calculate Euclidean distance
            const dist = Math.hypot(left.x - mirroredRight.x, left.y - mirroredRight.y);
            const maxDist = 0.15; // Normalized distance (15% of face width)
            const score = Math.max(0, 100 - (dist / maxDist * 100));
            scores.push(score);
        }
    });
    
    return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 50;
}

function calculateFwhr(points) {
    // Face width (zygomatic breadth) - indices 234 and 454
    if (points[234] && points[454]) {
        const faceWidth = Math.hypot(points[234].x - points[454].x, points[234].y - points[454].y);
        
        // Face height (nasion to gnathion) - indices 168 and 152
        const faceHeight = Math.hypot(points[168].x - points[152].x, points[168].y - points[152].y);
        
        if (faceHeight > 0) {
            const fwhr = faceWidth / faceHeight;
            const deviation = Math.abs(fwhr - IDEAL_RATIOS.fwhr);
            return Math.max(0, 100 - (deviation * 200));
        }
    }
    return 50;
}

function calculateJawlineScore(points) {
    // Calculate gonial angle (jaw angle) using indices 2, 8, 14
    if (points[2] && points[8] && points[14]) {
        const leftGonion = points[2];
        const rightGonion = points[14];
        const chin = points[8];
        
        // Vector calculations
        const v1 = { x: leftGonion.x - chin.x, y: leftGonion.y - chin.y };
        const v2 = { x: rightGonion.x - chin.x, y: rightGonion.y - chin.y };
        
        const mag1 = Math.hypot(v1.x, v1.y);
        const mag2 = Math.hypot(v2.x, v2.y);
        
        if (mag1 > 0 && mag2 > 0) {
            const dot = v1.x * v2.x + v1.y * v2.y;
            const cosAngle = dot / (mag1 * mag2);
            const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle))) * 180 / Math.PI;
            
            // Optimal gonial angle is 115-130 degrees
            if (angle >= 115 && angle <= 130) {
                return 100;
            } else if (angle < 115) {
                return 80 + (angle / 115 * 20);
            } else {
                return Math.max(0, 100 - ((angle - 130) / 30 * 100));
            }
        }
    }
    return 50;
}

function calculateBrowRidgeScore(points) {
    // Calculate brow ridge prominence using indices 70-77 (left brow) and 300-307 (right brow)
    let leftBrowY = 0, rightBrowY = 0;
    let leftCount = 0, rightCount = 0;
    
    for (let i = 70; i <= 77; i++) {
        if (points[i]) {
            leftBrowY += points[i].y;
            leftCount++;
        }
    }
    
    for (let i = 300; i <= 307; i++) {
        if (points[i]) {
            rightBrowY += points[i].y;
            rightCount++;
        }
    }
    
    if (leftCount > 0 && rightCount > 0 && points[33] && points[263]) {
        const avgBrowY = (leftBrowY / leftCount + rightBrowY / rightCount) / 2;
        const leftEyeY = points[33].y;
        const rightEyeY = points[263].y;
        const avgEyeY = (leftEyeY + rightEyeY) / 2;
        
        const browHeight = avgBrowY - avgEyeY; // Positive means brow above eye
        
        // Optimal brow ridge: prominent but not too heavy
        if (browHeight > 0.06) return 100;
        if (browHeight > 0.04) return 75 + ((browHeight - 0.04) / 0.02 * 25);
        return 50 + (browHeight / 0.04 * 25);
    }
    return 50;
}

function calculateChinScore(points) {
    // Calculate chin definition using indices 152 (chin), 17 (lower lip), 200 (neck)
    if (points[152] && points[17] && points[200]) {
        const chin = points[152];
        const lowerLip = points[17];
        const neck = points[200];
        
        // Chin projection (forward)
        const chinProjection = Math.abs(chin.x - lowerLip.x);
        
        // Chin angle
        const chinVector = { x: chin.x - lowerLip.x, y: chin.y - lowerLip.y };
        const verticalVector = { x: 0, y: 1 };
        
        const magChin = Math.hypot(chinVector.x, chinVector.y);
        
        if (magChin > 0) {
            const dot = chinVector.x * verticalVector.x + chinVector.y * verticalVector.y;
            const cosAngle = dot / magChin;
            const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle))) * 180 / Math.PI;
            
            const projScore = Math.min(100, chinProjection * 1000);
            const angleScore = 100 - Math.abs(angle - 90) / 90 * 100;
            
            return projScore * 0.4 + angleScore * 0.6;
        }
    }
    return 50;
}

function calculateEyeAreaScore(points) {
    // Calculate eye shape and symmetry using indices 33-133 (left eye) and 362-263 (right eye)
    const leftEyeIndices = [33, 133, 157, 158, 159, 160, 161, 173, 246];
    const rightEyeIndices = [362, 263, 387, 386, 385, 384, 398, 466];
    
    let leftEyePoints = leftEyeIndices.filter(i => points[i]).map(i => points[i]);
    let rightEyePoints = rightEyeIndices.filter(i => points[i]).map(i => points[i]);
    
    if (leftEyePoints.length < 3 || rightEyePoints.length < 3) return 50;
    
    // Calculate eye width/height ratios
    const leftWidth = Math.abs(points[33].x - points[133].x);
    const leftHeight = Math.abs(Math.max(...leftEyePoints.map(p => p.y)) - Math.min(...leftEyePoints.map(p => p.y)));
    const leftRatio = leftHeight / leftWidth;
    
    const rightWidth = Math.abs(points[362].x - points[263].x);
    const rightHeight = Math.abs(Math.max(...rightEyePoints.map(p => p.y)) - Math.min(...rightEyePoints.map(p => p.y)));
    const rightRatio = rightHeight / rightWidth;
    
    // Ideal eye shape ratio is around 0.3-0.4
    const leftScore = 100 - Math.abs(leftRatio - 0.35) * 200;
    const rightScore = 100 - Math.abs(rightRatio - 0.35) * 200;
    const symmetryScore = 100 - Math.abs(leftRatio - rightRatio) * 200;
    
    return (leftScore * 0.3 + rightScore * 0.3 + symmetryScore * 0.4);
}

function calculateNoseScore(points) {
    // Calculate nose shape using indices 1 (tip), 94 (left nostril), 279 (right nostril), 168 (bridge)
    if (points[1] && points[94] && points[279] && points[168]) {
        const noseTip = points[1];
        const leftNostril = points[94];
        const rightNostril = points[279];
        const noseBridge = points[168];
        
        // Nose width proportion
        const noseWidth = Math.abs(leftNostril.x - rightNostril.x);
        const faceWidth = Math.abs(points[234]?.x - points[454]?.x) || 1;
        const widthRatio = noseWidth / faceWidth;
        const widthScore = 100 - Math.abs(widthRatio - IDEAL_RATIOS.noseWidth) * 400;
        
        // Nose bridge straightness
        const bridgeSlope = Math.abs(noseBridge.y - noseTip.y) / (Math.abs(noseBridge.x - noseTip.x) + 0.001);
        const straightScore = bridgeSlope > 2 ? 100 : 50 + bridgeSlope * 25;
        
        return widthScore * 0.5 + straightScore * 0.5;
    }
    return 50;
}

function calculateLipScore(points) {
    // Calculate lip fullness using indices 61-291 (mouth corners), 37-267 (upper lip), 84-314 (lower lip)
    if (points[61] && points[291] && points[37] && points[267] && points[84] && points[314]) {
        const leftCorner = points[61];
        const rightCorner = points[291];
        const upperLip = (points[37].y + points[267].y) / 2;
        const lowerLip = (points[84].y + points[314].y) / 2;
        
        const lipWidth = Math.abs(leftCorner.x - rightCorner.x);
        const lipHeight = Math.abs(lowerLip - upperLip);
        
        if (lipWidth > 0) {
            const lipRatio = lipHeight / lipWidth;
            return 100 - Math.abs(lipRatio - IDEAL_RATIOS.lipFullness) * 200;
        }
    }
    return 50;
}

function calculateFacialThirdsScore(points) {
    // Calculate facial thirds harmony (forehead to brows, brows to nose base, nose base to chin)
    if (points[10] && points[70] && points[1] && points[152]) {
        const forehead = points[10].y;
        const brow = points[70].y;
        const noseBase = points[1].y;
        const chin = points[152].y;
        
        const third1 = Math.abs(brow - forehead);
        const third2 = Math.abs(noseBase - brow);
        const third3 = Math.abs(chin - noseBase);
        
        const avgThird = (third1 + third2 + third3) / 3;
        const deviations = [third1, third2, third3].map(t => Math.abs(t - avgThird) / avgThird);
        const avgDeviation = deviations.reduce((a, b) => a + b, 0) / 3;
        
        return Math.max(0, 100 - avgDeviation * 200);
    }
    return 50;
}

function calculateCanthalTilt(points) {
    // Calculate eye canthal tilt (positive tilt is attractive)
    if (points[133] && points[33] && points[362] && points[263]) {
        const leftInner = points[133];
        const leftOuter = points[33];
        const rightInner = points[362];
        const rightOuter = points[263];
        
        // Calculate tilt angles
        const leftAngle = Math.atan2(leftInner.y - leftOuter.y, leftInner.x - leftOuter.x) * 180 / Math.PI;
        const rightAngle = Math.atan2(rightInner.y - rightOuter.y, rightInner.x - rightOuter.x) * 180 / Math.PI;
        
        const avgAngle = (leftAngle + rightAngle) / 2;
        
        // Positive tilt (5-10 degrees) is ideal
        if (avgAngle > 5) {
            return Math.min(100, 80 + (avgAngle - 5) * 4);
        } else if (avgAngle > 0) {
            return 60 + (avgAngle / 5) * 20;
        } else {
            return Math.max(0, 60 + avgAngle * 6);
        }
    }
    return 50;
}

function calculateMidfaceRatio(points) {
    // Calculate midface ratio (pupil to mouth vs mouth to chin)
    if (points[468] && points[473] && points[13] && points[152]) {
        const leftPupil = points[468];
        const rightPupil = points[473];
        const mouthCenter = points[13];
        const chin = points[152];
        
        const pupilAvgY = (leftPupil.y + rightPupil.y) / 2;
        const pupilToMouth = Math.abs(mouthCenter.y - pupilAvgY);
        const mouthToChin = Math.abs(chin.y - mouthCenter.y);
        
        if (mouthToChin > 0) {
            const ratio = pupilToMouth / mouthToChin;
            return 100 - Math.abs(ratio - 1) * 50;
        }
    }
    return 50;
}

function calculateGonialAngle(points) {
    // Calculate gonial angle (jaw angle) - alternative calculation
    if (points[4] && points[2] && points[8]) {
        const ramus = points[4];
        const gonion = points[2];
        const chin = points[8];
        
        const v1 = { x: gonion.x - ramus.x, y: gonion.y - ramus.y };
        const v2 = { x: chin.x - gonion.x, y: chin.y - gonion.y };
        
        const mag1 = Math.hypot(v1.x, v1.y);
        const mag2 = Math.hypot(v2.x, v2.y);
        
        if (mag1 > 0 && mag2 > 0) {
            const dot = v1.x * v2.x + v1.y * v2.y;
            const cosAngle = dot / (mag1 * mag2);
            const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle))) * 180 / Math.PI;
            
            if (angle >= 115 && angle <= 135) {
                return 100;
            } else if (angle < 115) {
                return 70 + (angle / 115 * 30);
            } else {
                return 100 - ((angle - 135) / 45 * 100);
            }
        }
    }
    return 50;
}

function getLooksCategory(score) {
    for (const category of LOOKS_CATEGORIES) {
        if (score >= category.min) {
            return category.name;
        }
    }
    return 'Sub2';
}

function calculatePotential(metrics) {
    // Structural factors (hard to change)
    const structuralMetrics = ['jawline', 'browRidge', 'chin', 'gonialAngle'];
    const structuralScore = structuralMetrics.reduce((sum, m) => sum + (metrics[m] || 50), 0) / structuralMetrics.length;
    
    // Soft tissue factors (more improvable)
    const softMetrics = ['symmetry', 'eyeArea', 'lipFullness', 'noseShape'];
    const softScore = softMetrics.reduce((sum, m) => sum + (metrics[m] || 50), 0) / softMetrics.length;
    
    // Base potential weighted toward structure
    const basePotential = structuralScore * 0.6 + softScore * 0.4;
    
    // Improvement potential (max 20 points)
    const improvement = 20 * (1 - softScore / 100);
    
    return Math.min(100, Math.round(basePotential + improvement));
}

function analyzeFlawsStrengths(metrics) {
    const flaws = [];
    const strengths = [];
    
    const threshold = 70;
    const strengthThreshold = 80;
    
    const metricNames = {
        symmetry: 'Symmetry',
        fwhr: 'Face Ratio',
        jawline: 'Jawline',
        browRidge: 'Brow Ridge',
        chin: 'Chin',
        eyeArea: 'Eye Area',
        noseShape: 'Nose Shape',
        lipFullness: 'Lip Fullness',
        facialThirds: 'Facial Thirds',
        canthalTilt: 'Canthal Tilt',
        midfaceRatio: 'Midface Ratio',
        gonialAngle: 'Jaw Angle'
    };
    
    Object.entries(metrics).forEach(([key, value]) => {
        const name = metricNames[key] || key;
        if (value < threshold) {
            flaws.push(`Below average ${name} (${value.toFixed(1)})`);
        } else if (value > strengthThreshold) {
            strengths.push(`Excellent ${name} (${value.toFixed(1)})`);
        }
    });
    
    return {
        flaws: flaws.slice(0, 5),
        strengths: strengths.slice(0, 5)
    };
}

function generateRecommendations(metrics) {
    const recs = [];
    
    if (metrics.symmetry < 70) {
        recs.push({
            area: 'Facial Symmetry',
            priority: 'high',
            tip: 'Sleep on your back, practice facial exercises, and ensure proper posture to improve symmetry.'
        });
    }
    
    if (metrics.jawline < 65) {
        recs.push({
            area: 'Jawline Definition',
            priority: 'high',
            tip: 'Incorporate jaw exercises (mewing, chin tucks), reduce body fat, and consider gua sha techniques.'
        });
    }
    
    if (metrics.browRidge < 60) {
        recs.push({
            area: 'Brow Ridge',
            priority: 'medium',
            tip: 'Practice eyebrow raising exercises and consider microcurrent devices for muscle definition.'
        });
    }
    
    if (metrics.chin < 60) {
        recs.push({
            area: 'Chin Projection',
            priority: 'medium',
            tip: 'Practice chin exercises, maintain proper tongue posture (mewing), and reduce sodium intake.'
        });
    }
    
    if (metrics.eyeArea < 65) {
        recs.push({
            area: 'Eye Area',
            priority: 'medium',
            tip: 'Get adequate sleep (7-9 hours), reduce screen time, use cold compresses for under-eye bags.'
        });
    }
    
    if (metrics.noseShape < 60) {
        recs.push({
            area: 'Nose Shape',
            priority: 'low',
            tip: 'Practice nasal breathing exercises and consider facial yoga techniques.'
        });
    }
    
    if (metrics.lipFullness < 60) {
        recs.push({
            area: 'Lip Fullness',
            priority: 'low',
            tip: 'Stay hydrated, use lip masks, and practice lip exercises for definition.'
        });
    }
    
    // Always include general recommendations
    recs.push({
        area: 'Overall',
        priority: 'always',
        tip: 'Maintain healthy body fat (10-15%), stay hydrated, get 7-9 hours sleep, use SPF daily, and practice good skincare.'
    });
    
    return recs;
}

// UI Event Handlers
document.addEventListener('DOMContentLoaded', async () => {
    const uploadContainer = document.getElementById('uploadContainer');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const retakeBtn = document.getElementById('retakeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContainer = document.getElementById('resultsContainer');
    const errorContainer = document.getElementById('errorContainer');
    const errorText = document.getElementById('errorText');
    
    let currentImageData = null;
    let isAnalyzing = false;
    
    // Initialize Human library on page load
    await initHuman();
    
    // Upload handlers
    uploadContainer.addEventListener('click', () => {
        fileInput.click();
    });
    
    uploadContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadContainer.style.borderColor = 'var(--accent)';
        uploadContainer.style.background = 'rgba(59, 130, 246, 0.05)';
    });
    
    uploadContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadContainer.style.borderColor = 'var(--border)';
        uploadContainer.style.background = 'var(--bg-tertiary)';
    });
    
    uploadContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadContainer.style.borderColor = 'var(--border)';
        uploadContainer.style.background = 'var(--bg-tertiary)';
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    });
    
    function handleFile(file) {
        if (file.size > 16 * 1024 * 1024) {
            showError('File too large. Maximum size is 16MB.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImageData = e.target.result;
            previewImage.src = currentImageData;
            uploadContainer.style.display = 'none';
            previewContainer.style.display = 'block';
            resultsContainer.style.display = 'none';
            errorContainer.style.display = 'none';
        };
        reader.onerror = () => {
            showError('Failed to read file. Please try again.');
        };
        reader.readAsDataURL(file);
    }
    
    retakeBtn.addEventListener('click', () => {
        fileInput.value = '';
        previewContainer.style.display = 'none';
        uploadContainer.style.display = 'block';
        resultsContainer.style.display = 'none';
        errorContainer.style.display = 'none';
        currentImageData = null;
        isAnalyzing = false;
    });
    
    analyzeBtn.addEventListener('click', async () => {
        if (!currentImageData || isAnalyzing) return;
        
        isAnalyzing = true;
        analyzeBtn.disabled = true;
        loadingIndicator.style.display = 'block';
        errorContainer.style.display = 'none';
        
        try {
            // Create image element from data URL
            const img = new Image();
            img.src = currentImageData;
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
            });
            
            // Run analysis
            const result = await analyzeFace(img);
            displayResults(result);
            
        } catch (error) {
            showError(error.message || 'Analysis failed. Please try again with a clearer photo.');
        } finally {
            loadingIndicator.style.display = 'none';
            analyzeBtn.disabled = false;
            isAnalyzing = false;
        }
    });
    
    function showError(message) {
        errorText.textContent = message;
        errorContainer.style.display = 'flex';
        loadingIndicator.style.display = 'none';
    }
    
    function displayResults(data) {
        document.getElementById('categoryDisplay').textContent = data.category;
        document.getElementById('potentialDisplay').textContent = `Potential: ${data.potential}`;
        document.getElementById('overallScore').textContent = data.overall.toFixed(1);
        
        const metricsGrid = document.getElementById('metricsGrid');
        metricsGrid.innerHTML = '';
        
        const metricIcons = {
            symmetry: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
            fwhr: 'M3 15v4h4l10-10-4-4L3 15z M15 3l4 4',
            jawline: 'M12 2C8 2 4 5 4 9c0 4 8 13 8 13s8-9 8-13c0-4-4-7-8-7z',
            browRidge: 'M2 12h20M8 8v8M16 8v8',
            chin: 'M12 22c4 0 8-3 8-7 0-4-4-7-8-7s-8 3-8 7c0 4 4 7 8 7z',
            eyeArea: 'M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
            noseShape: 'M5 10c0-2 3-6 7-6s7 4 7 6c0 4-3 8-7 8s-7-4-7-8z',
            lipFullness: 'M3 14c0-2 2-5 5-5h8c3 0 5 3 5 5s-2 5-5 5H8c-3 0-5-3-5-5z',
            facialThirds: 'M4 4v16M10 4v16M16 4v16M20 4v16',
            canthalTilt: 'M8 8l4 4 4-4M8 16l4-4 4 4',
            midfaceRatio: 'M4 12h16M8 8v8M16 8v8',
            gonialAngle: 'M4 20L20 4M4 4l16 16'
        };
        
        Object.entries(data.metrics).forEach(([key, value]) => {
            const icon = metricIcons[key] || 'M12 2v20M2 12h20';
            const name = key.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());
            
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.setAttribute('data-tooltip', `${name}: ${value.toFixed(1)}`);
            card.innerHTML = `
                <div class="metric-header">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="${icon}"></path>
                    </svg>
                    ${name}
                </div>
                <div class="metric-value">${value.toFixed(1)}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${value}%"></div>
                </div>
            `;
            metricsGrid.appendChild(card);
        });
        
        const flawsList = document.getElementById('flawsList');
        flawsList.innerHTML = '';
        if (data.flaws.length > 0) {
            data.flaws.forEach(flaw => {
                const li = document.createElement('li');
                li.className = 'flaw';
                li.innerHTML = `<span class="flaw-text">${flaw}</span>`;
                flawsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.innerHTML = '<span class="flaw-text">No significant flaws detected</span>';
            flawsList.appendChild(li);
        }
        
        const strengthsList = document.getElementById('strengthsList');
        strengthsList.innerHTML = '';
        if (data.strengths.length > 0) {
            data.strengths.forEach(strength => {
                const li = document.createElement('li');
                li.className = 'strength';
                li.innerHTML = `<span class="strength-text">${strength}</span>`;
                strengthsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.innerHTML = '<span class="strength-text">Keep improving, potential is high</span>';
            strengthsList.appendChild(li);
        }
        
        const recommendationsList = document.getElementById('recommendationsList');
        recommendationsList.innerHTML = '';
        data.recommendations.forEach(rec => {
            const recItem = document.createElement('div');
            recItem.className = 'recommendation-item';
            
            let priorityClass = 'priority-medium';
            if (rec.priority === 'high') priorityClass = 'priority-high';
            if (rec.priority === 'low') priorityClass = 'priority-low';
            if (rec.priority === 'always') priorityClass = 'priority-always';
            
            recItem.innerHTML = `
                <span class="rec-priority ${priorityClass}">${rec.priority}</span>
                <div class="rec-content">
                    <div class="rec-area">${rec.area}</div>
                    <div class="rec-tip">${rec.tip}</div>
                </div>
            `;
            recommendationsList.appendChild(recItem);
        });
        
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    // Info button
    document.getElementById('infoBtn').addEventListener('click', () => {
        alert('Ascend uses advanced facial analysis to provide honest looksmaxxing ratings. All processing happens in your browser—no images are ever uploaded to any server.');
    });
    
    // Share button
    document.getElementById('shareBtn').addEventListener('click', () => {
        if (resultsContainer.style.display === 'block') {
            const text = `My Ascend results: ${document.getElementById('overallScore').textContent} (${document.getElementById('categoryDisplay').textContent})`;
            navigator.clipboard.writeText(text).then(() => {
                showError('Results copied to clipboard!');
                setTimeout(() => {
                    errorContainer.style.display = 'none';
                }, 2000);
            });
        } else {
            showError('Analyze a photo first to share results');
        }
    });
    
    // Theme toggle
    document.getElementById('themeBtn').addEventListener('click', () => {
        const root = document.documentElement;
        const currentBg = getComputedStyle(root).getPropertyValue('--bg-primary').trim();
        
        if (currentBg === '#0a0a0c') {
            root.style.setProperty('--bg-primary', '#f9fafb');
            root.style.setProperty('--bg-secondary', '#ffffff');
            root.style.setProperty('--bg-tertiary', '#f3f4f6');
            root.style.setProperty('--text-primary', '#111827');
            root.style.setProperty('--text-secondary', '#4b5563');
            root.style.setProperty('--text-tertiary', '#6b7280');
            root.style.setProperty('--border', '#e5e7eb');
            root.style.setProperty('--card-shadow', '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.05)');
        } else {
            root.style.setProperty('--bg-primary', '#0a0a0c');
            root.style.setProperty('--bg-secondary', '#111316');
            root.style.setProperty('--bg-tertiary', '#1a1d23');
            root.style.setProperty('--text-primary', '#ffffff');
            root.style.setProperty('--text-secondary', '#9ca3af');
            root.style.setProperty('--text-tertiary', '#6b7280');
            root.style.setProperty('--border', '#2a2f38');
            root.style.setProperty('--card-shadow', '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.3)');
        }
    });
});
