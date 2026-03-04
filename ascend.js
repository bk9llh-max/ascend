// ascend.js - Production Grade Facial Analysis Engine
// Version 3.0 - Optimized for Vercel Deployment
// Features: ML-enhanced, 3D validation, temporal smoothing, confidence scoring

// ==================== ADVANCED CONFIGURATION ====================
const ASCEND_CONFIG = {
    version: '3.0.0',
    environment: typeof window !== 'undefined' ? 'browser' : 'node',
    models: {
        basePath: 'https://cdn.jsdelivr.net/npm/@vladmandic/human@4.6.0/models/',
        cache: true,
        warmup: true,
        retryAttempts: 3,
        timeout: 30000
    },
    analysis: {
        minConfidence: 0.65,
        maxFaces: 1,
        temporalSmoothing: 0.25,
        useGPU: true,
        enable3D: true,
        enableIris: true,
        highPrecision: true
    },
    scoring: {
        weights: {
            symmetry: 0.18,
            fwhr: 0.14,
            jawline: 0.14,
            browRidge: 0.10,
            chin: 0.10,
            eyeArea: 0.10,
            noseShape: 0.06,
            lipFullness: 0.05,
            facialThirds: 0.05,
            canthalTilt: 0.03,
            midfaceRatio: 0.03,
            gonialAngle: 0.02
        },
        thresholds: {
            flaw: 70,
            strength: 80,
            critical: 50,
            elite: 95,
            excellent: 90
        }
    },
    cache: {
        enabled: true,
        maxSize: 50,
        ttl: 3600000 // 1 hour
    }
};

// ==================== SCIENTIFIC IDEALS ====================
const FACIAL_IDEALS = {
    // Based on peer-reviewed anthropometric studies
    ratios: {
        fwhr: { male: 0.75, female: 0.80, universal: 0.775 }, // Face width/height
        eyeSpacing: 0.46,  // 46% of face width
        noseWidth: 0.25,   // 25% of face width
        mouthWidth: 0.35,  // 35% of face width
        lipFullness: 0.35, // Height/width ratio
        midface: 1.0,      // Equal thirds
        chinProjection: 0.03 // Normalized
    },
    angles: {
        canthalTilt: 8,    // Degrees (positive tilt)
        gonialAngle: 125,  // Degrees (jaw angle)
        nasalAngle: 35,    // Degrees
        browAngle: 15      // Degrees
    },
    harmony: {
        goldenRatio: 1.618,
        facialThirds: [1.0, 1.0, 1.0], // Equal thirds
        phi: 1.618
    }
};

// ==================== LANDMARK INDICES ====================
const LANDMARK_INDICES = {
    // Face outline
    faceContour: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    
    // Eyes
    leftEye: [33,133,157,158,159,160,161,173,246],
    rightEye: [362,263,387,386,385,384,398,466],
    leftIris: [468,469,470,471,472],
    rightIris: [473,474,475,476,477],
    
    // Eyebrows
    leftBrow: [70,71,72,73,74,75,76,77],
    rightBrow: [300,301,302,303,304,305,306,307],
    
    // Nose
    noseBridge: [168,6,197,195,5,4,1,19],
    noseTip: [1],
    nostrils: [94,279],
    
    // Mouth
    mouthCorners: [61,291],
    upperLip: [61,185,40,39,37,0,267,269,270,409,291],
    lowerLip: [146,91,181,84,17,314,405,321,375,324],
    
    // Jaw
    jawLine: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    chin: [152],
    gonion: [2,14],
    
    // Key points
    pupils: [468,473],
    glabella: [168],
    nasion: [168],
    subnasale: [1]
};

// ==================== CACHE MANAGER ====================
class AnalysisCache {
    constructor(maxSize = 50, ttl = 3600000) {
        this.cache = new Map();
        this.maxSize = maxSize;
        this.ttl = ttl;
    }

    generateKey(imageData) {
        // Create hash of image data for caching
        let hash = 0;
        const str = imageData.substring(0, 1000); // Sample first 1000 chars
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return `analysis_${Math.abs(hash)}`;
    }

    set(key, value) {
        if (this.cache.size >= this.maxSize) {
            // Remove oldest entry
            const oldestKey = this.cache.keys().next().value;
            this.cache.delete(oldestKey);
        }
        this.cache.set(key, {
            data: value,
            timestamp: Date.now()
        });
    }

    get(key) {
        const item = this.cache.get(key);
        if (!item) return null;
        
        if (Date.now() - item.timestamp > this.ttl) {
            this.cache.delete(key);
            return null;
        }
        
        return item.data;
    }

    clear() {
        this.cache.clear();
    }
}

// ==================== MAIN ANALYZER CLASS ====================
class AscendAnalyzer {
    constructor() {
        this.human = null;
        this.modelsLoaded = false;
        this.cache = new AnalysisCache();
        this.temporalBuffer = [];
        this.lastResult = null;
        this.performanceMetrics = {
            initTime: 0,
            avgAnalysisTime: 0,
            totalAnalyses: 0
        };
    }

    async initialize() {
        const startTime = performance.now();
        
        for (let attempt = 1; attempt <= ASCEND_CONFIG.models.retryAttempts; attempt++) {
            try {
                console.log(`[Ascend] Initializing (attempt ${attempt}/${ASCEND_CONFIG.models.retryAttempts})...`);
                
                // Dynamic import for better performance
                const Human = (await import('https://cdn.jsdelivr.net/npm/@vladmandic/human@4.6.0/dist/human.esm.js')).default;
                
                this.human = new Human({
                    backend: ASCEND_CONFIG.analysis.useGPU ? 'webgl' : 'cpu',
                    async: true,
                    warmup: 'face',
                    cacheSensitivity: 0.95,
                    modelBasePath: ASCEND_CONFIG.models.basePath,
                    face: {
                        enabled: true,
                        detector: { 
                            enabled: true, 
                            maxDetected: ASCEND_CONFIG.analysis.maxFaces,
                            minConfidence: ASCEND_CONFIG.analysis.minConfidence,
                            model: 'blazeface',
                            return: true
                        },
                        mesh: { 
                            enabled: true, 
                            model: 'facemesh',
                            return: true
                        },
                        iris: { enabled: ASCEND_CONFIG.analysis.enableIris },
                        age: { enabled: true },
                        gender: { enabled: true }
                    },
                    body: { enabled: false },
                    hand: { enabled: false },
                    object: { enabled: false }
                });

                await this.human.load();
                
                // Warm up models
                await this.warmup();
                
                this.modelsLoaded = true;
                this.performanceMetrics.initTime = performance.now() - startTime;
                
                console.log(`[Ascend] Initialized successfully in ${Math.round(this.performanceMetrics.initTime)}ms`);
                return true;
                
            } catch (error) {
                console.error(`[Ascend] Attempt ${attempt} failed:`, error);
                if (attempt === ASCEND_CONFIG.models.retryAttempts) {
                    throw new Error(`Failed to initialize: ${error.message}`);
                }
                await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
            }
        }
    }

    async warmup() {
        try {
            const canvas = document.createElement('canvas');
            canvas.width = 128;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#808080';
            ctx.fillRect(0, 0, 128, 128);
            ctx.fillStyle = '#404040';
            ctx.fillRect(32, 32, 64, 64);
            
            await this.human.detect(canvas);
            console.log('[Ascend] Models warmed up');
        } catch (e) {
            // Silent fail for warmup
        }
    }

    async analyze(imageElement, options = {}) {
        const startTime = performance.now();
        
        try {
            if (!this.modelsLoaded) {
                await this.initialize();
            }

            // Check cache
            if (ASCEND_CONFIG.cache.enabled && imageElement.src) {
                const cacheKey = this.cache.generateKey(imageElement.src);
                const cached = this.cache.get(cacheKey);
                if (cached) {
                    console.log('[Ascend] Returning cached result');
                    return cached;
                }
            }

            // Preprocess image
            const processedImage = await this.preprocessImage(imageElement);
            
            // Run detection
            const result = await this.human.detect(processedImage);
            
            // Validate result
            this.validateResult(result);
            
            // Extract face data
            const face = result.face[0];
            const landmarks = face.mesh;
            const confidence = face.confidence;
            const gender = face.gender?.score > 0.65 ? face.gender.gender : 'unknown';
            const age = face.age || 25;
            
            // Calculate metrics
            const metrics = this.calculateMetrics(landmarks, gender);
            
            // Apply confidence weighting
            const confidenceWeight = Math.min(1, confidence / 0.8);
            Object.keys(metrics).forEach(key => {
                if (typeof metrics[key] === 'number') {
                    metrics[key] = metrics[key] * confidenceWeight;
                }
            });
            
            // Temporal smoothing
            if (ASCEND_CONFIG.analysis.temporalSmoothing > 0) {
                metrics = this.applyTemporalSmoothing(metrics);
            }
            
            // Calculate composite scores
            const overall = this.calculateOverallScore(metrics);
            const category = this.determineCategory(overall);
            const potential = this.calculatePotential(metrics, age, gender);
            const harmony = this.calculateHarmony(metrics);
            
            // Generate insights
            const flaws = this.identifyFlaws(metrics);
            const strengths = this.identifyStrengths(metrics);
            const recommendations = this.generateRecommendations(metrics, gender, age, flaws);
            
            // Build result
            const analysisResult = {
                success: true,
                metrics,
                overall: Math.round(overall * 10) / 10,
                category: category.name,
                categoryDescription: category.description,
                potential: Math.round(potential),
                harmony: Math.round(harmony * 10) / 10,
                flaws,
                strengths,
                recommendations,
                confidence: {
                    overall: Math.round(confidence * 100),
                    landmarks: Math.round((face.meshScore || 0.9) * 100),
                    detection: Math.round(confidence * 100)
                },
                performance: {
                    analysisTime: Math.round(performance.now() - startTime),
                    timestamp: Date.now()
                },
                metadata: {
                    gender,
                    age: Math.round(age),
                    faceSize: face.box ? Math.round(face.box.width * face.box.height) : 0,
                    landmarksCount: landmarks.length
                },
                version: ASCEND_CONFIG.version
            };
            
            // Cache result
            if (ASCEND_CONFIG.cache.enabled && imageElement.src) {
                const cacheKey = this.cache.generateKey(imageElement.src);
                this.cache.set(cacheKey, analysisResult);
            }
            
            // Update performance metrics
            this.performanceMetrics.totalAnalyses++;
            this.performanceMetrics.avgAnalysisTime = (
                this.performanceMetrics.avgAnalysisTime * (this.performanceMetrics.totalAnalyses - 1) + 
                (performance.now() - startTime)
            ) / this.performanceMetrics.totalAnalyses;
            
            return analysisResult;
            
        } catch (error) {
            console.error('[Ascend] Analysis error:', error);
            throw error;
        }
    }

    async preprocessImage(imageElement) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Calculate optimal dimensions
            let width = imageElement.width;
            let height = imageElement.height;
            const maxDim = 1024;
            
            if (width > maxDim || height > maxDim) {
                if (width > height) {
                    height = (height / width) * maxDim;
                    width = maxDim;
                } else {
                    width = (width / height) * maxDim;
                    height = maxDim;
                }
            }
            
            canvas.width = Math.round(width);
            canvas.height = Math.round(height);
            
            // Apply preprocessing for better detection
            ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
            
            // Apply subtle enhancements
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            // Gentle contrast enhancement
            for (let i = 0; i < data.length; i += 4) {
                // Convert to grayscale for edge detection
                const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                
                // Subtle sharpening
                if (i > canvas.width * 4 && i < data.length - canvas.width * 4) {
                    data[i] = Math.min(255, Math.max(0, data[i] + (data[i] - gray) * 0.2));
                    data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + (data[i + 1] - gray) * 0.2));
                    data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + (data[i + 2] - gray) * 0.2));
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
            resolve(canvas);
        });
    }

    validateResult(result) {
        if (!result || !result.face || result.face.length === 0) {
            throw new Error('No face detected. Please ensure:\n• Good lighting\n• Face is clearly visible\n• Front-facing position');
        }
        
        const face = result.face[0];
        
        if (!face.mesh || face.mesh.length < 400) {
            throw new Error('Insufficient facial landmarks detected. Please try a clearer photo.');
        }
        
        if (face.confidence < ASCEND_CONFIG.analysis.minConfidence) {
            throw new Error(`Low confidence detection (${Math.round(face.confidence * 100)}%). Please ensure better lighting.`);
        }
        
        // Check face orientation
        const mesh = face.mesh;
        const leftEye = mesh[33];
        const rightEye = mesh[263];
        
        if (leftEye && rightEye) {
            const eyeDistance = Math.abs(leftEye[0] - rightEye[0]);
            const faceWidth = Math.abs(mesh[234]?.[0] - mesh[454]?.[0] || eyeDistance * 2);
            
            if (eyeDistance / faceWidth < 0.25) {
                throw new Error('Face appears angled. Please use a front-facing photo.');
            }
        }
    }

    calculateMetrics(landmarks, gender) {
        const points = landmarks.map(p => ({ 
            x: p[0], 
            y: p[1], 
            z: p[2] || 0 
        }));
        
        // Use gender-specific ideals
        const ideals = { ...FACIAL_IDEALS.ratios };
        ideals.fwhr = gender === 'male' ? FACIAL_IDEALS.ratios.fwhr.male : 
                     gender === 'female' ? FACIAL_IDEALS.ratios.fwhr.female : 
                     FACIAL_IDEALS.ratios.fwhr.universal;
        
        return {
            symmetry: this.calculateSymmetry(points),
            fwhr: this.calculateFwhr(points, ideals.fwhr),
            jawline: this.calculateJawline(points),
            browRidge: this.calculateBrowRidge(points),
            chin: this.calculateChin(points),
            eyeArea: this.calculateEyeArea(points),
            noseShape: this.calculateNose(points, ideals),
            lipFullness: this.calculateLips(points, ideals),
            facialThirds: this.calculateFacialThirds(points),
            canthalTilt: this.calculateCanthalTilt(points),
            midfaceRatio: this.calculateMidfaceRatio(points),
            gonialAngle: this.calculateGonialAngle(points)
        };
    }

    calculateSymmetry(points) {
        const pairs = [
            { left: 33, right: 263, weight: 0.15 },
            { left: 133, right: 362, weight: 0.15 },
            { left: 70, right: 300, weight: 0.10 },
            { left: 77, right: 307, weight: 0.10 },
            { left: 61, right: 291, weight: 0.10 },
            { left: 37, right: 267, weight: 0.08 },
            { left: 84, right: 314, weight: 0.08 },
            { left: 94, right: 279, weight: 0.08 },
            { left: 2, right: 14, weight: 0.08 },
            { left: 4, right: 12, weight: 0.08 }
        ];
        
        let weightedScore = 0;
        let totalWeight = 0;
        
        pairs.forEach(({ left, right, weight }) => {
            if (points[left] && points[right]) {
                const lp = points[left];
                const rp = points[right];
                
                // Calculate mirror point
                const midX = (lp.x + rp.x) / 2;
                const mirrored = {
                    x: 2 * midX - rp.x,
                    y: rp.y,
                    z: rp.z
                };
                
                // 3D distance
                const dist = Math.sqrt(
                    Math.pow(lp.x - mirrored.x, 2) +
                    Math.pow(lp.y - mirrored.y, 2) +
                    Math.pow(lp.z - mirrored.z, 2)
                );
                
                // Normalize by face size
                const faceSize = Math.abs(points[10]?.x - points[152]?.x) || 1;
                const normDist = dist / faceSize;
                
                // Exponential scoring
                const score = 100 * Math.exp(-normDist * 8);
                
                weightedScore += score * weight;
                totalWeight += weight;
            }
        });
        
        return totalWeight > 0 ? weightedScore / totalWeight : 50;
    }

    calculateFwhr(points, ideal) {
        if (!points[234] || !points[454] || !points[168] || !points[152]) return 50;
        
        const leftZygomatic = points[234];
        const rightZygomatic = points[454];
        const nasion = points[168];
        const gnathion = points[152];
        
        const faceWidth = Math.hypot(
            leftZygomatic.x - rightZygomatic.x,
            leftZygomatic.y - rightZygomatic.y,
            leftZygomatic.z - rightZygomatic.z
        );
        
        const faceHeight = Math.hypot(
            nasion.x - gnathion.x,
            nasion.y - gnathion.y,
            nasion.z - gnathion.z
        );
        
        if (faceHeight === 0) return 50;
        
        const fwhr = faceWidth / faceHeight;
        const deviation = Math.abs(fwhr - ideal);
        
        // Gaussian scoring
        return 100 * Math.exp(-Math.pow(deviation / 0.06, 2));
    }

    calculateJawline(points) {
        if (!points[2] || !points[8] || !points[14]) return 50;
        
        const leftGonion = points[2];
        const rightGonion = points[14];
        const chin = points[8];
        
        // Calculate gonial angle
        const v1 = {
            x: leftGonion.x - chin.x,
            y: leftGonion.y - chin.y,
            z: leftGonion.z - chin.z
        };
        
        const v2 = {
            x: rightGonion.x - chin.x,
            y: rightGonion.y - chin.y,
            z: rightGonion.z - chin.z
        };
        
        const mag1 = Math.hypot(v1.x, v1.y, v1.z);
        const mag2 = Math.hypot(v2.x, v2.y, v2.z);
        
        if (mag1 === 0 || mag2 === 0) return 50;
        
        const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        const cosAngle = dot / (mag1 * mag2);
        const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle))) * 180 / Math.PI;
        
        // Score gonial angle
        let angleScore;
        if (angle >= 115 && angle <= 135) {
            angleScore = 100;
        } else if (angle < 115) {
            angleScore = 60 + (angle / 115) * 40;
        } else {
            angleScore = 100 - ((angle - 135) / 45) * 100;
        }
        
        // Calculate jaw curvature
        const jawPoints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            .filter(i => points[i])
            .map(i => points[i]);
        
        let curvatureScore = 70;
        if (jawPoints.length > 5) {
            const yValues = jawPoints.map(p => p.y);
            const meanY = yValues.reduce((a, b) => a + b, 0) / yValues.length;
            const variance = yValues.reduce((a, b) => a + Math.pow(b - meanY, 2), 0) / yValues.length;
            curvatureScore = Math.min(100, Math.sqrt(variance) * 800);
        }
        
        return angleScore * 0.7 + curvatureScore * 0.3;
    }

    calculateBrowRidge(points) {
        const leftBrow = [70,71,72,73,74,75,76,77];
        const rightBrow = [300,301,302,303,304,305,306,307];
        
        let leftZ = 0, rightZ = 0;
        let leftCount = 0, rightCount = 0;
        
        leftBrow.forEach(i => {
            if (points[i]) {
                leftZ += points[i].z;
                leftCount++;
            }
        });
        
        rightBrow.forEach(i => {
            if (points[i]) {
                rightZ += points[i].z;
                rightCount++;
            }
        });
        
        if (leftCount === 0 || rightCount === 0 || !points[33] || !points[263]) return 50;
        
        const avgBrowZ = (leftZ / leftCount + rightZ / rightCount) / 2;
        const leftEyeZ = points[33].z;
        const rightEyeZ = points[263].z;
        const avgEyeZ = (leftEyeZ + rightEyeZ) / 2;
        
        const prominence = avgBrowZ - avgEyeZ;
        
        // Normalize
        const faceDepth = Math.abs(points[168]?.z - points[152]?.z) || 1;
        const normalized = prominence / faceDepth;
        
        if (normalized > 0.07) return 100;
        if (normalized > 0.04) return 70 + (normalized - 0.04) * 1000;
        if (normalized > 0.02) return 50 + (normalized - 0.02) * 1000;
        return 50;
    }

    calculateChin(points) {
        if (!points[152] || !points[17] || !points[200]) return 50;
        
        const chin = points[152];
        const lowerLip = points[17];
        const neck = points[200];
        
        // Chin projection
        const projection = Math.abs(chin.x - lowerLip.x);
        
        // Chin angle
        const chinVector = {
            x: chin.x - lowerLip.x,
            y: chin.y - lowerLip.y,
            z: chin.z - lowerLip.z
        };
        
        const vertical = { x: 0, y: 1, z: 0 };
        
        const mag = Math.hypot(chinVector.x, chinVector.y, chinVector.z);
        
        if (mag === 0) return 50;
        
        const dot = chinVector.x * vertical.x + chinVector.y * vertical.y + chinVector.z * vertical.z;
        const cosAngle = dot / mag;
        const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle))) * 180 / Math.PI;
        
        const projScore = Math.min(100, projection * 800);
        const angleScore = 100 - Math.abs(angle - 90) / 90 * 100;
        
        return projScore * 0.5 + angleScore * 0.5;
    }

    calculateEyeArea(points) {
        const leftEye = [33,133,157,158,159,160,161,173,246];
        const rightEye = [362,263,387,386,385,384,398,466];
        
        const leftPoints = leftEye.filter(i => points[i]).map(i => points[i]);
        const rightPoints = rightEye.filter(i => points[i]).map(i => points[i]);
        
        if (leftPoints.length < 5 || rightPoints.length < 5) return 50;
        
        // Eye opening ratios
        const leftWidth = Math.hypot(points[33].x - points[133].x, points[33].y - points[133].y);
        const leftHeight = Math.abs(Math.max(...leftPoints.map(p => p.y)) - Math.min(...leftPoints.map(p => p.y)));
        const leftRatio = leftHeight / leftWidth;
        
        const rightWidth = Math.hypot(points[362].x - points[263].x, points[362].y - points[263].y);
        const rightHeight = Math.abs(Math.max(...rightPoints.map(p => p.y)) - Math.min(...rightPoints.map(p => p.y)));
        const rightRatio = rightHeight / rightWidth;
        
        // Optimal ratio 0.3-0.4
        const leftScore = 100 - Math.abs(leftRatio - 0.35) * 200;
        const rightScore = 100 - Math.abs(rightRatio - 0.35) * 200;
        const symmetryScore = 100 - Math.abs(leftRatio - rightRatio) * 200;
        
        // Check for hooded eyes
        const leftHooded = points[70] && points[33] ? 
            Math.max(0, (points[70].y - points[33].y) * 500) : 0;
        const rightHooded = points[300] && points[362] ? 
            Math.max(0, (points[300].y - points[362].y) * 500) : 0;
        
        const hoodedPenalty = (leftHooded + rightHooded) / 2;
        
        return Math.max(0, 
            leftScore * 0.2 + 
            rightScore * 0.2 + 
            symmetryScore * 0.3 + 
            50 - hoodedPenalty * 0.3
        );
    }

    calculateNose(points, ideals) {
        if (!points[1] || !points[94] || !points[279] || !points[168]) return 50;
        
        const noseTip = points[1];
        const leftNostril = points[94];
        const rightNostril = points[279];
        const noseBridge = points[168];
        
        // Width ratio
        const faceWidth = Math.abs(points[234]?.x - points[454]?.x) || 1;
        const noseWidth = Math.abs(leftNostril.x - rightNostril.x);
        const widthRatio = noseWidth / faceWidth;
        const widthScore = 100 - Math.abs(widthRatio - ideals.noseWidth) * 400;
        
        // Bridge straightness
        const bridgePoints = [168,6,197,195,5,4,1]
            .filter(i => points[i])
            .map(i => points[i]);
        
        let straightScore = 70;
        if (bridgePoints.length > 2) {
            const xVals = bridgePoints.map(p => p.x);
            const meanX = xVals.reduce((a, b) => a + b, 0) / xVals.length;
            const deviations = bridgePoints.map(p => Math.abs(p.x - meanX));
            const avgDev = deviations.reduce((a, b) => a + b, 0) / deviations.length;
            straightScore = 100 - Math.min(100, avgDev * 200);
        }
        
        // Projection
        const projection = Math.abs(noseTip.x - noseBridge.x);
        const projScore = Math.min(100, projection * 500);
        
        return widthScore * 0.4 + straightScore * 0.4 + projScore * 0.2;
    }

    calculateLips(points, ideals) {
        if (!points[61] || !points[291] || !points[37] || !points[267]) return 50;
        
        const leftCorner = points[61];
        const rightCorner = points[291];
        const upperLip = (points[37].y + points[267].y) / 2;
        const lowerLip = points[84] && points[314] ? 
            (points[84].y + points[314].y) / 2 : upperLip + 0.05;
        
        const lipWidth = Math.hypot(leftCorner.x - rightCorner.x, leftCorner.y - rightCorner.y);
        const lipHeight = Math.abs(lowerLip - upperLip);
        
        if (lipWidth === 0) return 50;
        
        const lipRatio = lipHeight / lipWidth;
        const ratioScore = 100 - Math.abs(lipRatio - ideals.lipFullness) * 200;
        
        // Cupid's bow
        let cupidScore = 60;
        if (points[0] && points[37] && points[267]) {
            const bowHeight = Math.abs(points[0].y - ((points[37].y + points[267].y) / 2));
            cupidScore = Math.min(100, bowHeight * 1000);
        }
        
        return ratioScore * 0.6 + cupidScore * 0.4;
    }

    calculateFacialThirds(points) {
        if (!points[10] || !points[70] || !points[1] || !points[152]) return 50;
        
        const forehead = points[10].y;
        const brow = points[70].y;
        const noseBase = points[1].y;
        const chin = points[152].y;
        
        const third1 = Math.abs(brow - forehead);
        const third2 = Math.abs(noseBase - brow);
        const third3 = Math.abs(chin - noseBase);
        
        const avg = (third1 + third2 + third3) / 3;
        
        const deviations = [
            Math.abs(third1 - avg) / avg,
            Math.abs(third2 - avg) / avg,
            Math.abs(third3 - avg) / avg
        ];
        
        const avgDeviation = deviations.reduce((a, b) => a + b, 0) / 3;
        
        return Math.max(0, 100 - avgDeviation * 150);
    }

    calculateCanthalTilt(points) {
        if (!points[133] || !points[33] || !points[362] || !points[263]) return 50;
        
        const leftInner = points[133];
        const leftOuter = points[33];
        const rightInner = points[362];
        const rightOuter = points[263];
        
        const leftAngle = Math.atan2(
            leftInner.y - leftOuter.y,
            leftInner.x - leftOuter.x
        ) * 180 / Math.PI;
        
        const rightAngle = Math.atan2(
            rightInner.y - rightOuter.y,
            rightInner.x - rightOuter.x
        ) * 180 / Math.PI;
        
        const avgAngle = (leftAngle + rightAngle) / 2;
        
        if (avgAngle >= 5 && avgAngle <= 12) return 100;
        if (avgAngle > 0) return 70 + (avgAngle / 12) * 30;
        return Math.max(0, 70 + avgAngle * 7);
    }

    calculateMidfaceRatio(points) {
        if (!points[468] || !points[473] || !points[13] || !points[152]) return 50;
        
        const pupilAvgY = (points[468].y + points[473].y) / 2;
        const mouthY = points[13].y;
        const chinY = points[152].y;
        
        const pupilToMouth = Math.abs(mouthY - pupilAvgY);
        const mouthToChin = Math.abs(chinY - mouthY);
        
        if (mouthToChin === 0) return 50;
        
        const ratio = pupilToMouth / mouthToChin;
        
        return 100 - Math.abs(ratio - 1) * 50;
    }

    calculateGonialAngle(points) {
        if (!points[4] || !points[2] || !points[8]) return 50;
        
        const ramus = points[4];
        const gonion = points[2];
        const chin = points[8];
        
        const v1 = {
            x: gonion.x - ramus.x,
            y: gonion.y - ramus.y,
            z: gonion.z - ramus.z
        };
        
        const v2 = {
            x: chin.x - gonion.x,
            y: chin.y - gonion.y,
            z: chin.z - gonion.z
        };
        
        const mag1 = Math.hypot(v1.x, v1.y, v1.z);
        const mag2 = Math.hypot(v2.x, v2.y, v2.z);
        
        if (mag1 === 0 || mag2 === 0) return 50;
        
        const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        const cosAngle = dot / (mag1 * mag2);
        const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle))) * 180 / Math.PI;
        
        if (angle >= 115 && angle <= 135) return 100;
        if (angle < 115) return 60 + (angle / 115) * 40;
        return 100 - ((angle - 135) / 45) * 100;
    }

    applyTemporalSmoothing(newMetrics) {
        this.temporalBuffer.push(newMetrics);
        if (this.temporalBuffer.length > 5) {
            this.temporalBuffer.shift();
        }
        
        if (this.temporalBuffer.length < 2) return newMetrics;
        
        const smoothed = {};
        const keys = Object.keys(newMetrics);
        
        keys.forEach(key => {
            if (typeof newMetrics[key] === 'number') {
                const avg = this.temporalBuffer.reduce((sum, m) => sum + (m[key] || 0), 0) / this.temporalBuffer.length;
                smoothed[key] = newMetrics[key] * (1 - ASCEND_CONFIG.analysis.temporalSmoothing) + 
                               avg * ASCEND_CONFIG.analysis.temporalSmoothing;
            } else {
                smoothed[key] = newMetrics[key];
            }
        });
        
        return smoothed;
    }

    calculateOverallScore(metrics) {
        let total = 0;
        let weightSum = 0;
        
        Object.entries(ASCEND_CONFIG.scoring.weights).forEach(([key, weight]) => {
            if (metrics[key] !== undefined) {
                total += metrics[key] * weight;
                weightSum += weight;
            }
        });
        
        return weightSum > 0 ? total / weightSum : 50;
    }

    calculateHarmony(metrics) {
        const values = Object.values(metrics).filter(v => typeof v === 'number');
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        
        return 100 - Math.min(100, variance);
    }

    calculatePotential(metrics, age, gender) {
        const structural = ['jawline', 'browRidge', 'chin', 'gonialAngle', 'fwhr']
            .reduce((sum, m) => sum + (metrics[m] || 50), 0) / 5;
        
        const soft = ['symmetry', 'eyeArea', 'lipFullness', 'noseShape', 'canthalTilt']
            .reduce((sum, m) => sum + (metrics[m] || 50), 0) / 5;
        
        let ageFactor;
        if (age < 20) ageFactor = 1.0;
        else if (age < 25) ageFactor = 0.9 - (age - 20) * 0.02;
        else if (age < 35) ageFactor = 0.8 - (age - 25) * 0.015;
        else ageFactor = 0.65 - (age - 35) * 0.01;
        
        ageFactor = Math.max(0.3, Math.min(1.0, ageFactor));
        
        const genderFactor = gender === 'male' ? 1.05 : 1.0;
        
        const base = structural * 0.5 + soft * 0.5;
        const improvement = 25 * (1 - soft / 100) * ageFactor * genderFactor;
        const ceiling = Math.min(98, structural + 20);
        
        return Math.min(ceiling, base + improvement);
    }

    determineCategory(score) {
        if (score >= 98) return { name: 'CHAD PREMIUM', description: 'Elite genetic lottery winner' };
        if (score >= 95) return { name: 'CHAD', description: 'Top 1% facial structure' };
        if (score >= 92) return { name: 'HTN+', description: 'High tier normie plus' };
        if (score >= 89) return { name: 'HTN', description: 'High tier normie' };
        if (score >= 86) return { name: 'HTN-', description: 'High tier normie minus' };
        if (score >= 83) return { name: 'MTN+', description: 'Mid tier normie plus' };
        if (score >= 80) return { name: 'MTN', description: 'Mid tier normie' };
        if (score >= 77) return { name: 'MTN-', description: 'Mid tier normie minus' };
        if (score >= 74) return { name: 'LTN+', description: 'Low tier normie plus' };
        if (score >= 71) return { name: 'LTN', description: 'Low tier normie' };
        if (score >= 68) return { name: 'LTN-', description: 'Low tier normie minus' };
        if (score >= 65) return { name: 'Sub5+', description: 'Below average plus' };
        if (score >= 62) return { name: 'Sub5', description: 'Below average' };
        if (score >= 59) return { name: 'Sub5-', description: 'Below average minus' };
        if (score >= 55) return { name: 'Sub4', description: 'Significant improvement needed' };
        if (score >= 50) return { name: 'Sub3', description: 'Major improvement needed' };
        return { name: 'Sub2', description: 'Consult specialist' };
    }

    identifyFlaws(metrics) {
        const flaws = [];
        const threshold = ASCEND_CONFIG.scoring.thresholds.flaw;
        
        const descriptions = {
            symmetry: 'Facial asymmetry',
            fwhr: 'Suboptimal face proportions',
            jawline: 'Weak jawline definition',
            browRidge: 'Underdeveloped brow ridge',
            chin: 'Chin projection needs improvement',
            eyeArea: 'Eye area could be enhanced',
            noseShape: 'Nose shape could be refined',
            lipFullness: 'Lips lack fullness',
            facialThirds: 'Unbalanced facial thirds',
            canthalTilt: 'Eye tilt is suboptimal',
            midfaceRatio: 'Midface proportions off',
            gonialAngle: 'Jaw angle needs work'
        };
        
        Object.entries(metrics).forEach(([key, value]) => {
            if (value < threshold && descriptions[key]) {
                flaws.push({
                    feature: key,
                    description: descriptions[key],
                    severity: value < 50 ? 'critical' : value < 60 ? 'major' : 'moderate',
                    score: Math.round(value * 10) / 10
                });
            }
        });
        
        return flaws.sort((a, b) => a.score - b.score).slice(0, 5);
    }

    identifyStrengths(metrics) {
        const strengths = [];
        const threshold = ASCEND_CONFIG.scoring.thresholds.strength;
        
        const descriptions = {
            symmetry: 'Excellent facial symmetry',
            fwhr: 'Ideal face proportions',
            jawline: 'Strong jawline definition',
            browRidge: 'Well-developed brow ridge',
            chin: 'Good chin projection',
            eyeArea: 'Attractive eye area',
            noseShape: 'Well-proportioned nose',
            lipFullness: 'Full, defined lips',
            facialThirds: 'Balanced facial thirds',
            canthalTilt: 'Attractive eye tilt',
            midfaceRatio: 'Harmonious midface',
            gonialAngle: 'Ideal jaw angle'
        };
        
        Object.entries(metrics).forEach(([key, value]) => {
            if (value > threshold && descriptions[key]) {
                strengths.push({
                    feature: key,
                    description: descriptions[key],
                    excellence: value > 90 ? 'exceptional' : 'good',
                    score: Math.round(value * 10) / 10
                });
            }
        });
        
        return strengths.sort((a, b) => b.score - a.score).slice(0, 5);
    }

    generateRecommendations(metrics, gender, age, flaws) {
        const recommendations = [];
        
        const recMap = {
            symmetry: {
                priority: 'high',
                tip: 'Sleep on back, practice facial exercises, check for dental issues, consider myofunctional therapy'
            },
            fwhr: {
                priority: 'high',
                tip: 'Optimize body fat (10-15% for males, 18-23% for females), facial exercises, mewing technique'
            },
            jawline: {
                priority: 'high',
                tip: 'Jaw exercises, mewing, body fat reduction, chewing gum, consider gua sha techniques'
            },
            browRidge: {
                priority: 'medium',
                tip: 'Brow exercises, microcurrent devices, eyebrow grooming, consider derma rolling'
            },
            chin: {
                priority: 'medium',
                tip: 'Chin exercises, proper tongue posture, mewing, reduce sodium intake'
            },
            eyeArea: {
                priority: 'medium',
                tip: 'Sleep 7-9 hours, cold compresses, reduce screen time, eye exercises, adequate hydration'
            },
            noseShape: {
                priority: 'low',
                tip: 'Nasal breathing exercises, facial yoga, nose sculpting exercises'
            },
            lipFullness: {
                priority: 'low',
                tip: 'Hydration, lip masks, lip exercises, gentle exfoliation'
            }
        };
        
        flaws.forEach(flaw => {
            if (recMap[flaw.feature] && flaw.severity !== 'exceptional') {
                recommendations.push({
                    area: flaw.feature.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase()),
                    priority: recMap[flaw.feature].priority,
                    tip: recMap[flaw.feature].tip,
                    basedOn: `${flaw.score} score`
                });
            }
        });
        
        // Always include general recommendations
        recommendations.push({
            area: 'General Health',
            priority: 'always',
            tip: `Maintain healthy body fat (${gender === 'male' ? '10-15%' : '18-23%'}), stay hydrated (2-3L daily), sleep 7-9 hours, use SPF daily, exercise regularly`
        });
        
        recommendations.push({
            area: 'Skincare',
            priority: 'always',
            tip: 'Daily moisturizer, vitamin C serum, retinol (night), SPF 50+ daily, gentle cleansing'
        });
        
        return recommendations.slice(0, 5);
    }
}

// ==================== GLOBAL INSTANCE ====================
const analyzer = new AscendAnalyzer();

// ==================== EXPORT FOR MODULES ====================
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AscendAnalyzer, analyzer, ASCEND_CONFIG };
} else {
    window.ascendAnalyzer = analyzer;
    window.AscendAnalyzer = AscendAnalyzer;
}
