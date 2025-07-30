// ========== Enhanced Vector3 Class ==========
class Vector3 {
  constructor(x = 0, y = 0, z = 0) { 
    this.x = x; 
    this.y = y; 
    this.z = z; 
  }

  add(v) { return new Vector3(this.x + v.x, this.y + v.y, this.z + v.z); }
  subtract(v) { return new Vector3(this.x - v.x, this.y - v.y, this.z - v.z); }
  multiply(s) { return new Vector3(this.x * s, this.y * s, this.z * s); }
  
  dot(v) { return this.x * v.x + this.y * v.y + this.z * v.z; }
  cross(v) {
    return new Vector3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }
  
  normalize() {
    const len = this.length();
    return len > 0 ? this.multiply(1 / len) : new Vector3();
  }
  
  lerp(v, t) {
    return this.add(v.subtract(this).multiply(t));
  }
  
  distanceTo(v) {
    const dx = this.x - v.x, dy = this.y - v.y, dz = this.z - v.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  
  length() { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z); }
  
  clone() { return new Vector3(this.x, this.y, this.z); }
  
  equals(v, tolerance = 0.001) {
    return Math.abs(this.x - v.x) < tolerance && 
           Math.abs(this.y - v.y) < tolerance && 
           Math.abs(this.z - v.z) < tolerance;
  }
}

// ========== Advanced Kalman Filter ==========
class AdvancedKalmanFilter {
  constructor(R = 0.01, Q = 0.09, processNoise = 0.1) {
    this.R = R; // Measurement noise
    this.Q = Q; // Process noise  
    this.processNoise = processNoise;
    
    // State: [position, velocity]
    this.state = [0, 0];
    this.covariance = [[1, 0], [0, 1]];
    this.isInitialized = false;
    this.lastTime = Date.now();
  }

  predict(dt) {
    if (!this.isInitialized) return;
    
    // State transition matrix
    const F = [[1, dt], [0, 1]];
    
    // Predict state
    const newState = [
      this.state[0] + this.state[1] * dt,
      this.state[1]
    ];
    
    // Predict covariance
    const newCov = [
      [this.covariance[0][0] + dt * this.covariance[1][0] + this.processNoise, 
       this.covariance[0][1] + dt * this.covariance[1][1]],
      [this.covariance[1][0], 
       this.covariance[1][1] + this.processNoise]
    ];
    
    this.state = newState;
    this.covariance = newCov;
  }

  update(measurement) {
    const currentTime = Date.now();
    const dt = (currentTime - this.lastTime) / 1000;
    this.lastTime = currentTime;
    
    if (!this.isInitialized) {
      this.state = [measurement, 0];
      this.isInitialized = true;
      return measurement;
    }
    
    this.predict(dt);
    
    // Kalman gain
    const S = this.covariance[0][0] + this.R;
    const K = [this.covariance[0][0] / S, this.covariance[1][0] / S];
    
    // Update state
    const residual = measurement - this.state[0];
    this.state[0] += K[0] * residual;
    this.state[1] += K[1] * residual;
    
    // Update covariance
    const newCov = [
      [(1 - K[0]) * this.covariance[0][0], 
       (1 - K[0]) * this.covariance[0][1]],
      [this.covariance[1][0] - K[1] * this.covariance[0][0], 
       this.covariance[1][1] - K[1] * this.covariance[0][1]]
    ];
    
    this.covariance = newCov;
    
    return this.state[0];
  }
  
  getPredictedPosition(timeAhead) {
    if (!this.isInitialized) return 0;
    return this.state[0] + this.state[1] * timeAhead;
  }
  
  getVelocity() {
    return this.isInitialized ? this.state[1] : 0;
  }

  reset() {
    this.state = [0, 0];
    this.covariance = [[1, 0], [0, 1]];
    this.isInitialized = false;
    this.lastTime = Date.now();
  }
}

// ========== Enhanced Drag Lock with Physics ==========
class EnhancedDragLock {
  constructor(config) {
    this.config = config;
    this.velocity = new Vector3();
    this.lastPosition = new Vector3();
    this.lastTime = Date.now();
  }

  applyDragLock(currentPos, targetPos, deltaTime) {
    const delta = targetPos.subtract(currentPos);
    const distance = delta.length();
    
    // Dynamic force based on distance
    let force = this.config.dragForce;
    if (distance > this.config.maxDistance) {
      force *= this.config.longRangeMultiplier;
    }
    
    // Apply momentum and smoothing
    const targetVelocity = delta.multiply(force);
    this.velocity = this.velocity.lerp(targetVelocity, this.config.smoothingFactor);
    
    // Apply velocity with time compensation
    const movement = this.velocity.multiply(deltaTime);
    const next = currentPos.add(movement);

    // Smart snapping with velocity consideration
    if (distance < this.config.snapThreshold && 
        this.velocity.length() < this.config.velocityThreshold &&
        this.config.enableSnap) {
      return targetPos;
    }

    this.lastPosition = next.clone();
    return next;
  }
  
  applyRecoilCompensation(currentPos, recoilVector) {
    if (!this.config.recoilCompensation) return currentPos;
    
    const compensation = recoilVector.multiply(-this.config.recoilFactor);
    return currentPos.add(compensation);
  }
}

// ========== Target Prediction System ==========
class TargetPredictor {
  constructor(config) {
    this.config = config;
    this.positionHistory = [];
    this.maxHistorySize = 10;
  }
  
  addPosition(position, timestamp) {
    this.positionHistory.push({ position: position.clone(), time: timestamp });
    if (this.positionHistory.length > this.maxHistorySize) {
      this.positionHistory.shift();
    }
  }
  
  predictPosition(timeAhead) {
    if (this.positionHistory.length < 2) {
      return this.positionHistory[0]?.position || new Vector3();
    }
    
    // Calculate average velocity
    const recent = this.positionHistory.slice(-3);
    let avgVelocity = new Vector3();
    
    for (let i = 1; i < recent.length; i++) {
      const dt = (recent[i].time - recent[i-1].time) / 1000;
      if (dt > 0) {
        const velocity = recent[i].position.subtract(recent[i-1].position).multiply(1/dt);
        avgVelocity = avgVelocity.add(velocity);
      }
    }
    
    avgVelocity = avgVelocity.multiply(1 / (recent.length - 1));
    
    // Predict future position
    const currentPos = this.positionHistory[this.positionHistory.length - 1].position;
    return currentPos.add(avgVelocity.multiply(timeAhead));
  }
  
  reset() {
    this.positionHistory = [];
  }
}

// ========== Enhanced Target Manager ==========
class EnhancedTargetManager {
  constructor(config) {
    this.config = config;
    this.targets = [];
    this.currentTarget = null;
    this.targetLockTime = 0;
    this.switchCooldown = 0;
  }

  addTarget(position, priority = 1, targetType = 'head') {
    const target = {
      id: Date.now() + Math.random(),
      position: position.clone(),
      priority,
      targetType,
      lastSeen: Date.now(),
      isVisible: true,
      health: 100,
      distance: 0
    };
    this.targets.push(target);
    return target;
  }
  
  updateTarget(targetId, newPosition) {
    const target = this.targets.find(t => t.id === targetId);
    if (target) {
      target.position = newPosition.clone();
      target.lastSeen = Date.now();
      target.isVisible = true;
    }
  }
  
  removeTarget(targetId) {
    this.targets = this.targets.filter(t => t.id !== targetId);
    if (this.currentTarget && this.currentTarget.id === targetId) {
      this.currentTarget = null;
    }
  }

  getBestTarget(cameraPos) {
    if (this.targets.length === 0) return null;
    
    const currentTime = Date.now();
    
    // Filter valid targets
    const validTargets = this.targets.filter(target => {
      const timeSinceLastSeen = currentTime - target.lastSeen;
      return target.isVisible && 
             timeSinceLastSeen < this.config.targetTimeout &&
             target.health > 0;
    });
    
    if (validTargets.length === 0) return null;
    
    // Update distances
    validTargets.forEach(target => {
      target.distance = cameraPos.distanceTo(target.position);
    });
    
    // Smart target selection algorithm
    let bestTarget = validTargets[0];
    let bestScore = this.calculateTargetScore(bestTarget, cameraPos);
    
    for (let i = 1; i < validTargets.length; i++) {
      const score = this.calculateTargetScore(validTargets[i], cameraPos);
      if (score > bestScore) {
        bestScore = score;
        bestTarget = validTargets[i];
      }
    }
    
    // Target switching logic
    if (this.currentTarget && this.currentTarget.id !== bestTarget.id) {
      if (this.switchCooldown > 0 && !this.config.instantSwitch) {
        return this.currentTarget;
      }
      this.switchCooldown = this.config.switchDelay;
    }
    
    this.currentTarget = bestTarget;
    return bestTarget;
  }
  
  calculateTargetScore(target, cameraPos) {
    let score = target.priority * 100;
    
    // Distance factor (closer is better)
    const maxDistance = this.config.maxTargetDistance || 50;
    const distanceFactor = Math.max(0, 1 - (target.distance / maxDistance));
    score += distanceFactor * 50;
    
    // Head target bonus
    if (target.targetType === 'head' && this.config.headLockOnly) {
      score += 30;
    }
    
    // Current target bonus (sticky targeting)
    if (this.currentTarget && this.currentTarget.id === target.id) {
      score += 20;
    }
    
    return score;
  }
  
  update(deltaTime) {
    if (this.switchCooldown > 0) {
      this.switchCooldown -= deltaTime;
    }
  }
}

// ========== Performance Monitor ==========
class PerformanceMonitor {
  constructor() {
    this.frameCount = 0;
    this.lastFpsUpdate = Date.now();
    this.fps = 0;
    this.frameTime = 0;
    this.lastFrameTime = Date.now();
  }
  
  startFrame() {
    this.lastFrameTime = Date.now();
  }
  
  endFrame() {
    const currentTime = Date.now();
    this.frameTime = currentTime - this.lastFrameTime;
    this.frameCount++;
    
    if (currentTime - this.lastFpsUpdate > 1000) {
      this.fps = this.frameCount;
      this.frameCount = 0;
      this.lastFpsUpdate = currentTime;
    }
  }
  
  getStats() {
    return {
      fps: this.fps,
      frameTime: this.frameTime,
      avgFrameTime: this.frameTime
    };
  }
}

// ========== Enhanced Camera ==========
class EnhancedCamera {
  constructor() {
    this.position = new Vector3();
    this.rotation = new Vector3();
    this.fov = 90;
    this.sensitivity = 1.0;
  }

  setPosition(vec) {
    this.position = vec.clone();
  }
  
  setRotation(rotation) {
    this.rotation = rotation.clone();
  }
  
  smoothMove(targetPos, deltaTime, smoothingFactor) {
    this.position = this.position.lerp(targetPos, smoothingFactor * deltaTime);
  }
  
  getViewDirection() {
    // Convert rotation to direction vector (simplified)
    const pitch = this.rotation.x * Math.PI / 180;
    const yaw = this.rotation.y * Math.PI / 180;
    
    return new Vector3(
      Math.cos(pitch) * Math.sin(yaw),
      -Math.sin(pitch),
      Math.cos(pitch) * Math.cos(yaw)
    );
  }
}

// ========== Smart Auto Fire ==========
class SmartAutoFire {
  constructor(config) {
    this.config = config;
    this.lastFireTime = 0;
    this.consecutiveHits = 0;
    this.accuracy = 10.0;
  }

  canFire(targetDistance, aimAccuracy) {
    const currentTime = Date.now();
    const timeSinceLastFire = currentTime - this.lastFireTime;
    
    // Rate limiting
    if (timeSinceLastFire < this.config.fireRate) return false;
    
    // Distance check
    if (targetDistance > this.config.maxFireDistance) return false;
    
    // Accuracy check
    if (aimAccuracy < this.config.minAccuracy) return false;
    
    return true;
  }

  fire(target, aimAccuracy) {
    if (!this.canFire(target.distance, aimAccuracy)) return false;
    
    this.lastFireTime = Date.now();
    
    // Simulate hit/miss based on accuracy
    const hitChance = aimAccuracy * this.accuracy;
    const hit = Math.random() < hitChance;
    
    if (hit) {
      this.consecutiveHits++;
      console.log(`[FIRE] 🎯 HIT! Target eliminated (${this.consecutiveHits} consecutive)`);
    } else {
      this.consecutiveHits = 0;
      console.log(`[FIRE] ❌ Miss (accuracy: ${(aimAccuracy*100).toFixed(1)}%)`);
    }
    
    return hit;
  }
  
  updateAccuracy(recentPerformance) {
    // Adaptive accuracy based on recent performance
    this.accuracy = Math.max(0.1, Math.min(1.0, recentPerformance));
  }
}

// ========== Enhanced Aim Lock Engine ==========
class EnhancedAimLockEngine {
  constructor(config) {
    this.config = config;
    this.isActive = false;
    this.lastUpdateTime = Date.now();
    this.updateInterval = null; // Store interval reference
    
    // Core systems
    this.kalmanX = new AdvancedKalmanFilter(config.kalman.R, config.kalman.Q);
    this.kalmanY = new AdvancedKalmanFilter(config.kalman.R, config.kalman.Q);
    this.kalmanZ = new AdvancedKalmanFilter(config.kalman.R, config.kalman.Q);
    
    this.dragLock = new EnhancedDragLock(config);
    this.camera = new EnhancedCamera();
    this.autoFire = new SmartAutoFire(config);
    this.targetManager = new EnhancedTargetManager(config);
    this.predictor = new TargetPredictor(config);
    this.performance = new PerformanceMonitor();
    
    // State tracking
    this.currentAccuracy = 0;
    this.lockDuration = 0;
    this.totalShots = 0;
    this.totalHits = 0;
  }
addTarget(position, priority = 1, boneType = 'head') {
  const pos = position instanceof Vector3
    ? position
    : new Vector3(position.x, position.y, position.z);
  
  const target = {
    id: Date.now() + Math.random(),
    position: pos,
    priority,
    bone: boneType,
    timestamp: Date.now()
  }

  this.targetManager.targets.push(target);
  this.targetManager.targets.sort((a, b) => b.priority - a.priority);

  if (this.targetManager.targets.length > this.config.maxTargets) {
    this.targetManager.targets.length = this.config.maxTargets;
  }

  return target;
}
  updateAimLock() {
    this.performance.startFrame();
    
    const currentTime = Date.now();
    const deltaTime = (currentTime - this.lastUpdateTime) / 1000;
    this.lastUpdateTime = currentTime;
    
    // Get best target
    const target = this.targetManager.getBestTarget(this.camera.position);
    if (!target) {
      this.resetTracking();
      this.performance.endFrame();
      return;
    }
    
    // Update prediction system
    this.predictor.addPosition(target.position, currentTime);
    
    // Apply Kalman filtering
    const filteredPos = new Vector3(
      this.kalmanX.update(target.position.x),
      this.kalmanY.update(target.position.y),
      this.kalmanZ.update(target.position.z)
    );
    
    // Predict future position
    const predictionTime = this.config.maxPredictionTime / 1000;
    const predictedPos = this.predictor.predictPosition(predictionTime);
    
    // Blend filtered and predicted positions
    const blendFactor = Math.min(target.distance / 20, 1);
    const finalTarget = filteredPos.lerp(predictedPos, blendFactor * this.config.predictionWeight);
    
    // Apply drag lock with physics
    const newPos = this.dragLock.applyDragLock(this.camera.position, finalTarget, deltaTime);
    this.camera.setPosition(newPos);
    
    // Calculate accuracy
    const aimError = newPos.distanceTo(finalTarget);
    this.currentAccuracy = Math.max(0, 1 - (aimError / this.config.snapThreshold));
    
    // Update lock duration
    if (this.currentAccuracy > 0.8) {
      this.lockDuration += deltaTime;
    } else {
      this.lockDuration = 0;
    }
    
    // Auto fire logic
    if (this.config.fireOnLock && this.lockDuration > this.config.minLockTime) {
      const fired = this.autoFire.fire(target, this.currentAccuracy);
      if (fired) {
        this.totalShots++;
        if (this.currentAccuracy > 0.9) this.totalHits++;
      }
    }
    
    // Update systems
    this.targetManager.update(deltaTime);
    this.autoFire.updateAccuracy(this.getHitRate());
    
    // Logging
    if (currentTime % 1000 < 16) { // Every ~1 second
      this.logStatus();
    }
    
    this.performance.endFrame();
  }
  
  resetTracking() {
    this.kalmanX.reset();
    this.kalmanY.reset();
    this.kalmanZ.reset();
    this.predictor.reset();
    this.lockDuration = 0;
  }
  
  getHitRate() {
    return this.totalShots > 0 ? this.totalHits / this.totalShots : 0;
  }
  
  logStatus() {
    const stats = this.performance.getStats();
    const hitRate = (this.getHitRate() * 100).toFixed(1);
    const accuracy = (this.currentAccuracy * 100).toFixed(1);
    
    console.log(`[ENGINE] FPS: ${stats.fps} | Accuracy: ${accuracy}% | Hit Rate: ${hitRate}% | Targets: ${this.targetManager.targets.length}`);
  }
  
  addTarget(x, y, z, priority = 1, type = 'head') {
    return this.targetManager.addTarget(new Vector3(x, y, z), priority, type);
  }
  
  start() {
    this.isActive = true;
    this.lastUpdateTime = Date.now();
    
    // Use setInterval instead of requestAnimationFrame for Node.js compatibility
    this.updateInterval = setInterval(() => {
      if (this.isActive) {
        this.updateAimLock();
      } else {
        clearInterval(this.updateInterval);
      }
    }, 1000 / this.config.maxFPS); // Use maxFPS from config (144 FPS = ~6.9ms)
    
    console.log(`[ENGINE] Enhanced Aim Lock Engine started at ${this.config.maxFPS} FPS`);
  }
  
  stop() {
    this.isActive = false;
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    console.log('[ENGINE] Enhanced Aim Lock Engine stopped');
  }
  
  getStats() {
    return {
      performance: this.performance.getStats(),
      accuracy: this.currentAccuracy,
      hitRate: this.getHitRate(),
      lockDuration: this.lockDuration,
      totalTargets: this.targetManager.targets.length,
      activeTarget: this.targetManager.currentTarget?.id || null
    };
  }
}

// ========== Enhanced Configuration ==========
const EnhancedConfig = {
  // Sensitivity settings
  aimSensitivity: 4.5,
  dragForce: 5.0,
  snapThreshold: 0.0014,
  velocityThreshold: 0.1,
  maxDistance: 99999,
  longRangeMultiplier: 1.2,
  
  // Prediction settings
  maxPredictionTime: 150,
  predictionWeight: 0.3,
  
  // Kalman filter
  kalman: { R: 0.004, Q: 0.01 },
  
  // Smoothing
  smoothingFactor: 0.001,
  
  // Recoil compensation
  recoilCompensation: true,
  recoilFactor: 0.8,
  
  // Target management
  lockMode: 'smart',
  headLockOnly: true,
  instantSwitch: false,
  switchDelay: 200,
  targetTimeout: 2000,
  maxTargetDistance: 99999,
  
  // Snap settings
  enableSnap: true,
  snapSpeed: 5.0,
  
  // Auto fire
  fireOnLock: true,
  minLockTime: 0.01,
  fireRate: 60, // ms between shots
  maxFireDistance: 99999,
  minAccuracy: 0.0,
  
  // Performance
  maxFPS: 144,
  adaptiveQuality: true
};

// ========== Demo Setup ==========
function createDemoScenario() {
  const engine = new EnhancedAimLockEngine(EnhancedConfig);

  // Add multiple targets using position (Vector3 or { x, y, z })
  const target1 = engine.addTarget(new Vector3(-0.045697,  -0.004478, 0.020043), 10, 'neck');   // High priority head
  const target2 = engine.addTarget({ x: -0.045697, y: -0.004478, z: -0.020043 }, 10, 'head');   // Lower priority body
  const target3 = engine.addTarget(new Vector3(-0.05334, -0.003515, -0.000763), 1, 'hips');   // Medium priority head

  // Simulate moving targets
  let time = 0;
  setInterval(() => {
    time += 0.016; // ~60 FPS simulation

    if (engine.targetManager && engine.targetManager.targets.length > 0) {
      const t1 = engine.targetManager.targets[0];
      if (t1) {
        t1.position.x = 2.5 + Math.sin(time) * 0.5;
        t1.position.y = 1.2 + Math.cos(time * 0.7) * 0.3;
      }

      const t2 = engine.targetManager.targets[1];
      if (t2) {
        t2.position.x = -1.8 + Math.cos(time * 1.2) * 0.8;
        t2.position.z = 1.5 + Math.sin(time * 0.8) * 0.4;
      }
    }
  }, 16);

  return engine;
}
// ========== Initialize Enhanced System ==========
console.log('🚀 Starting Enhanced Aim Lock Engine...');
const enhancedEngine = createDemoScenario();
enhancedEngine.start();

// Performance monitoring
setInterval(() => {
  const stats = enhancedEngine.getStats();
  console.log(`📊 Performance: ${JSON.stringify(stats, null, 2)}`);
}, 5000);
