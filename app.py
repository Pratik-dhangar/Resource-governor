import streamlit as st
import psutil
import pandas as pd
import numpy as np
import time
import pickle
import os
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
MODEL_FILE = 'sentinai_model.pkl'
SCALER_FILE = 'sentinai_scaler.pkl'
WINDOW_SIZE = 5  # Seconds to wait before confirming anomaly
CONTAMINATION = 0.05 

st.set_page_config(page_title="SentinAI: Advanced Guardian", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'history_cpu' not in st.session_state:
    st.session_state['history_cpu'] = deque(maxlen=60)
if 'anomaly_buffer' not in st.session_state:
    st.session_state['anomaly_buffer'] = deque(maxlen=WINDOW_SIZE)
# Session-based model storage (unique per user session)
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

# --- HELPER: IDENTIFY TOP PROCESS ---
def get_top_process():
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                pinfo = proc.info
                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        sorted_procs = sorted(processes, key=lambda p: p['cpu_percent'], reverse=True)
        if sorted_procs:
            return sorted_procs[0]
        return None
    except Exception:
        return None

# --- HELPER: GET METRICS ---
def get_metrics():
    cpu = psutil.cpu_percent(interval=0.0)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_io_counters()
    disk_activity = (disk.read_bytes + disk.write_bytes) / 1024 / 1024 # MB
    return [cpu, ram, disk_activity]

# --- UI HEADER ---
st.title("🛡️ SentinAI: Process-Aware Resource Governor")
st.markdown("Advanced Edge AI that detects resource anomalies and identifies the **Process ID** responsible.")
st.info("ℹ️ This system learns your normal PC behavior and alerts you when unusual resource spikes occur, pinpointing the exact process causing the issue.")

col_nav1, col_nav2 = st.columns([1, 3])

with col_nav1:
    st.subheader("Control System")
    st.caption("Switch between monitoring your system or training the AI on your usage patterns.")
    mode = st.radio("System Mode:", ["Monitor (Active)", "Retrain System"])
    
    if st.session_state['model_trained']:
        st.success("✅ Your Personal AI Model Loaded")
        if st.button("🗑️ Reset Model", help="Clear your trained model and start fresh"):
            st.session_state['model'] = None
            st.session_state['scaler'] = None
            st.session_state['model_trained'] = False
            st.success("Model reset! Please retrain.")
            st.rerun()
    else:
        st.warning("⚠️ No Model Found. Please Train First.")

# --- MODE: RETRAIN SYSTEM ---
if mode == "Retrain System":
    st.header("🧠 Context Learning Phase")
    st.info("Click Start. For 30 seconds, vary your usage (Idle vs Active).")
    st.caption("The AI will observe your CPU, RAM, and Disk activity to understand what's normal for your system.")
    
    if st.button("Start Context Learning"):
        progress = st.progress(0)
        status = st.empty()
        training_data = []
        
        for i in range(30):
            m = get_metrics()
            training_data.append(m)
            status.text(f"Learning Context... CPU: {m[0]}% | RAM: {m[1]}% | Disk: {m[2]:.1f}MB")
            progress.progress((i+1)/30)
            time.sleep(1)
            
        # Data Science Pipeline
        # We explicitly name columns here
        df = pd.DataFrame(training_data, columns=['CPU', 'RAM', 'Disk'])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        
        model = IsolationForest(contamination=CONTAMINATION, random_state=42)
        model.fit(X_scaled)
        
        # Store in session state (user-specific, in-memory)
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['model_trained'] = True
            
        st.success("✅ Training Complete! Your personal model is ready. Switch to 'Monitor' mode.")

# --- MODE: MONITOR (ACTIVE) ---
elif mode == "Monitor (Active)":
    if not st.session_state['model_trained']:
        st.error("Please train your personal model first.")
        st.stop()
        
    # Load from session state (user-specific model)
    model = st.session_state['model']
    scaler = st.session_state['scaler']
        
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1_metric = kpi1.empty()
    kpi2_metric = kpi2.empty()
    status_metric = kpi3.empty()
    
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Live Tensor Stream")
        st.caption("Real-time CPU usage over the last 60 seconds - spikes indicate potential anomalies.")
        chart_place = st.empty()
    with c2:
        st.subheader("Forensics Log")
        st.caption("AI analysis results - shows which process is causing resource anomalies.")
        log_place = st.empty()
        
    start_monitoring = st.checkbox("Enable Real-Time Watchdog", value=True)
    st.caption("Toggle to start/stop continuous monitoring. The AI scans your system every second.")
    
    if start_monitoring:
        while True:
            # 1. ACQUIRE DATA
            raw_metrics = get_metrics() 
            
            # --- THE FIX IS HERE ---
            # We convert the list to a DataFrame with NAMES to satisfy Sklearn warning
            input_df = pd.DataFrame([raw_metrics], columns=['CPU', 'RAM', 'Disk'])
            
            # 2. PREPROCESS
            vector_scaled = scaler.transform(input_df)
            
            # 3. INFERENCE
            prediction = model.predict(vector_scaled)[0]
            score = model.decision_function(vector_scaled)[0]
            
            # 4. SLIDING WINDOW LOGIC
            st.session_state['anomaly_buffer'].append(prediction)
            anomaly_count = st.session_state['anomaly_buffer'].count(-1)
            # Require majority (3 out of 5) for confirmed anomaly
            is_confirmed_anomaly = anomaly_count >= 3
            
            # 5. FORENSICS
            culprit_name = "System"
            culprit_pid = "N/A"
            
            if is_confirmed_anomaly:
                top_proc = get_top_process()
                if top_proc:
                    culprit_name = top_proc['name']
                    culprit_pid = top_proc['pid']
            
            # 6. UI UPDATE
            kpi1_metric.metric("CPU Load", f"{raw_metrics[0]}%")
            kpi2_metric.metric("RAM Usage", f"{raw_metrics[1]}%")
            
            if is_confirmed_anomaly:
                status_metric.metric("Status", "CRITICAL ANOMALY", delta="-ALERT", delta_color="inverse")
                log_place.error(f"🚨 **ANOMALY DETECTED**\n\n**Culprit:** {culprit_name}\n**PID:** {culprit_pid}\n**CPU:** {raw_metrics[0]}%\n**Confidence:** {abs(score):.2f}")
            elif prediction == -1:
                status_metric.metric("Status", "Analyzing Spike...", delta="⚠️", delta_color="off")
                log_place.warning("⚠️ Short spike detected (filtering noise...)")
            else:
                status_metric.metric("Status", "System Nominal", delta="✓", delta_color="normal")
                log_place.success("✅ System running normally")
            
            st.session_state['history_cpu'].append(raw_metrics[0])
            
            # Create DataFrame with proper labels for chart
            cpu_data = pd.DataFrame(
                list(st.session_state['history_cpu']),
                columns=["CPU Usage (%)"]
            )
            chart_place.line_chart(cpu_data, x_label="Time (seconds)", y_label="CPU (%)")
            
            time.sleep(1)