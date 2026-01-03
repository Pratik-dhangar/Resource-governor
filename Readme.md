# 🛡️ SentinAI: Process-Aware Resource Governor
<div align="center">  <a href="https://resource-governor-obqqqkh768wsbssggcbzds.streamlit.app/" target="_blank"><img src="https://img.shields.io/badge/🚀Live-Demo-success?style=for-the-badge" alt="Live Demo"></a> </div>
An **Edge AI-powered system monitor** that learns your computer's normal behavior and detects resource anomalies in real-time. When unusual CPU, RAM, or Disk activity is detected, SentinAI automatically identifies the exact process responsible.

## 🎯 What Makes This "Edge AI"?

Unlike cloud-based monitoring solutions, SentinAI runs **entirely on your local machine**:

- **No Internet Required**: All AI inference happens locally
- **Privacy-First**: Your system metrics never leave your device
- **Ultra-Low Latency**: Detects anomalies in under 1 second
- **Resource Efficient**: Lightweight model (~10KB) with minimal overhead
- **Personalized**: Learns YOUR unique usage patterns, not generic benchmarks

## 🌟 Features

- **Real-Time Anomaly Detection**: Continuously monitors CPU, RAM, and Disk I/O every second
- **Process Forensics**: Automatically identifies which application (PID & name) is causing resource spikes
- **Smart Filtering**: Uses a sliding window approach to ignore brief spikes and confirm sustained anomalies
- **Self-Learning**: Trains on your actual usage patterns to minimize false positives
- **Session-Based Models**: Each user gets their own personalized model (perfect for multi-user deployments)
- **Interactive Dashboard**: Built with Streamlit for live visualization and control

## 🧠 Edge AI Concepts Used

### 1. **Unsupervised Learning (Isolation Forest)**

Traditional monitoring tools require you to set static thresholds (e.g., "alert if CPU > 80%"). SentinAI uses **Isolation Forest**, an anomaly detection algorithm that:

- **Learns Normal Patterns**: Understands what's "normal" for YOUR system (e.g., video editing might use 90% CPU regularly)
- **Detects Outliers**: Identifies unusual combinations of CPU + RAM + Disk activity
- **No Labels Required**: Works without pre-defined "good" vs "bad" examples

**How it Works:**
```python
# Build decision trees that "isolate" anomalies
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(normal_usage_data)

# Predict: -1 = Anomaly, 1 = Normal
prediction = model.predict(current_metrics)
```

### 2. **Feature Scaling (StandardScaler)**

Raw metrics vary wildly in scale:
- CPU: 0-100%
- RAM: 0-100%
- Disk I/O: 0-1000+ MB/s

Without scaling, the model would be biased toward Disk activity. SentinAI uses **StandardScaler** to normalize all features:

```python
# Transform all metrics to mean=0, std=1
scaler = StandardScaler()
scaled_data = scaler.fit_transform([cpu, ram, disk])
```

This ensures the AI treats a 10% CPU spike with the same importance as a 100MB/s disk spike.

### 3. **Temporal Filtering (Sliding Window)**

A single-frame anomaly might just be a momentary spike (e.g., opening a file). SentinAI uses a **5-second sliding window**:

```python
anomaly_buffer = deque(maxlen=5)  # Last 5 seconds
anomaly_buffer.append(prediction)

# Only trigger alert if 4 out of 5 seconds show anomaly
confirmed = anomaly_buffer.count(-1) >= 4
```

This prevents "crying wolf" on harmless, transient spikes.

### 4. **Edge Deployment Strategy**

**Session-Based Model Storage**: For multi-user deployments (like Streamlit Cloud), models are stored in `st.session_state` instead of disk files:

```python
# Store model in user's browser session (in-memory)
st.session_state['model'] = model
st.session_state['scaler'] = scaler
st.session_state['model_trained'] = True
```

**Why Session Storage?**
- ✅ **Privacy**: Each user's model is isolated from others
- ✅ **Personalization**: User A's gaming PC model doesn't affect User B's laptop model
- ✅ **No Conflicts**: Concurrent users don't overwrite each other's data
- ⚠️ **Trade-off**: Model resets when browser session ends (requires retraining on revisit)

**Inference Pipeline**:
1. **Acquire** → Read CPU/RAM/Disk from `psutil`
2. **Preprocess** → Scale using session-stored scaler
3. **Infer** → Predict anomaly (< 50ms)
4. **Act** → Identify culprit process if anomaly confirmed

### 5. **Contextual Learning**

Instead of using a generic "one-size-fits-all" model, SentinAI performs **context calibration**:

- Observes YOUR system for 30 seconds during both idle and active states
- Learns YOUR baseline (gaming PC vs. lightweight laptop)
- Creates a personalized model stored locally

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Resource-governor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   # OR
   python -m streamlit run app.py
   ```

## 🚀 Usage

### First Time Setup (Training)

1. Switch to **"Retrain System"** mode
2. Click **"Start Context Learning"**
3. For 30 seconds, use your computer normally:
   - Spend ~15 seconds idle
   - Spend ~15 seconds actively using apps (browse, type, open files)
4. Wait for training to complete
5. Switch to **"Monitor (Active)"** mode

### Continuous Monitoring

1. Select **"Monitor (Active)"** mode
2. Enable **"Real-Time Watchdog"** checkbox
3. The dashboard will now:
   - Display live CPU/RAM metrics every second
   - Show a 60-second rolling graph of CPU usage
   - Alert you when anomalies are detected
   - Identify the exact process causing the spike

### When to Retrain

Retrain your model if:
- You've upgraded hardware
- Your typical usage patterns change significantly
- You're getting too many false positives

## 📊 Dashboard Overview

### Control Panel
- **System Mode**: Switch between monitoring and training
- **Brain Status**: Shows if a trained model is loaded

### Live Telemetry
- **CPU Load**: Current CPU usage percentage
- **RAM Usage**: Current memory consumption
- **Status**: System state (Nominal / Analyzing / Critical Anomaly)

### Live Tensor Stream
Real-time line chart showing CPU usage over the last 60 seconds

### Forensics Log
When an anomaly is detected, displays:
- Process name causing the issue
- Process ID (PID)
- CPU usage at detection time
- Confidence score

## 📂 Project Structure

```
Resource-governor/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

**Note**: Models are stored in-memory per session (`st.session_state`), not as persistent `.pkl` files.

## 🔬 Technical Details

### Dependencies
- **streamlit**: Web dashboard framework
- **psutil**: System monitoring (CPU, RAM, Disk)
- **scikit-learn**: Machine learning (Isolation Forest, StandardScaler)
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### Model Specifications
- **Algorithm**: Isolation Forest
- **Contamination**: 5% (assumes 5% of data points are anomalies)
- **Features**: 3 (CPU %, RAM %, Disk MB/s)
- **Training Time**: ~30 seconds
- **Inference Time**: <50ms per prediction
- **Model Size**: ~10KB
- **Storage**: In-memory session state (browser-specific, not persisted to disk)

### Performance

- **Training Required**: Must calibrate to your specific system
- **Session-Based**: Model resets when you close your browser (requires retraining on next visit)
- **Single-Machine**: Doesn't correlate anomalies across multiple devices
- **Resource Intensive Apps**: May flag legitimate heavy workloads (e.g., video rendering)
- **No Historical Analysis**: Only monitors current state, doesn't log long-term trends




