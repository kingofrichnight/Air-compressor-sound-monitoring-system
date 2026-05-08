# Air Compressor Sound Monitoring System (IoT + ML)

## Overview

This project is an Industrial IoT (IIoT) based air compressor sound monitoring system developed using a Raspberry Pi, sound sensor, machine learning, MTConnect, MySQL, and Grafana.

The system continuously captures compressor sound data in real time, processes the audio signal, extracts sound features such as RMS values and Mel spectrograms, and classifies the compressor state into:

* OFF
* REST / IDLE
* RUNNING

The classified results are then transmitted through MTConnect and stored in a MySQL database for visualization on a Grafana dashboard.

---

# Project Objectives

* Monitor air compressor operating conditions using sound
* Detect compressor states automatically
* Reduce manual monitoring effort
* Provide real-time industrial dashboard visualization
* Store historical data for analysis and maintenance
* Demonstrate Industrial IoT integration using machine learning

---

# System Architecture

```text
Microphone / Sound Sensor
          ↓
     Raspberry Pi
          ↓
 Audio Processing (Python)
          ↓
 Feature Extraction
(RMS + Mel Spectrogram)
          ↓
 CNN Machine Learning Model
          ↓
 Compressor State Prediction
          ↓
 MTConnect Adapter & Agent
          ↓
 MySQL Database
          ↓
 Grafana Dashboard
```

---

# Technologies Used

| Category             | Technology               |
| -------------------- | ------------------------ |
| Programming Language | Python                   |
| Audio Processing     | librosa, numpy, scipy    |
| Machine Learning     | TensorFlow / Keras CNN   |
| Real-Time Data       | MTConnect                |
| Database             | MySQL / MariaDB          |
| Dashboard            | Grafana                  |
| Hardware             | Raspberry Pi             |
| Audio Format         | WAV (48 kHz, 16-bit PCM) |

---

# Hardware Requirements

## Required Components

* Raspberry Pi
* USB microphone or industrial sound sensor
* SD card with Raspberry Pi OS
* Internet or local network connection
* Computer for dashboard and database

---

# Software Requirements

Install the following Python libraries:

```bash
pip install numpy librosa matplotlib tensorflow requests pymysql
```

Optional:

```bash
pip install mtconnect
```

---

# Dataset Structure

The dataset is organized into folders based on compressor state.

```text
Dataset/
│
├── OFF/
├── REST/
└── RUNNING/
```

Each folder contains WAV audio files recorded under different operating conditions.

---

# Audio Specifications

| Parameter        | Value         |
| ---------------- | ------------- |
| Sampling Rate    | 48,000 Hz     |
| Audio Type       | Mono          |
| Bit Depth        | 16-bit PCM    |
| File Format      | .wav          |
| Recording Length | 10–30 seconds |

---

# Feature Extraction

The system extracts audio features from WAV files.

## Features Used

* RMS (Root Mean Square)
* Mel Spectrogram
* Amplitude Information
* Frequency Distribution

## RMS Threshold Examples

| Compressor State | Approximate RMS Range |
| ---------------- | --------------------- |
| OFF              | 0.0009 – 0.025        |
| REST             | 0.01 – 0.10           |
| RUNNING          | 0.50 – 0.88           |

These values may vary depending on environment noise and microphone distance.

---

# Machine Learning Model

## Model Used

Convolutional Neural Network (CNN)

## Model Workflow

1. Load WAV audio
2. Convert to Mel spectrogram
3. Normalize data
4. Train CNN model
5. Predict compressor state

## Output Classes

* OFF
* REST
* RUNNING

---

# Real-Time Monitoring

The Raspberry Pi continuously:

1. Collects sound samples
2. Processes incoming audio
3. Extracts features
4. Runs ML prediction
5. Sends state data to MTConnect
6. Stores results in MySQL
7. Displays live dashboard in Grafana

---

# Database Integration

The predicted state and sound measurements are stored in MySQL.

Example fields:

| Field            | Description         |
| ---------------- | ------------------- |
| timestamp        | Time of measurement |
| sound_level      | Sound intensity     |
| rms_value        | RMS amplitude       |
| compressor_state | Predicted state     |

---

# Grafana Dashboard

Grafana is used to visualize:

* Real-time compressor state
* RMS trends
* Sound levels
* Historical compressor activity
* System monitoring data

---

# MTConnect Integration

MTConnect is used for industrial communication.

The system sends:

* Compressor state
* Sound data
* RMS measurements
* Execution status

This enables interoperability with industrial monitoring systems.

---

# Example Workflow

```text
Audio Input
   ↓
Feature Extraction
   ↓
CNN Prediction
   ↓
State Classification
   ↓
Database Storage
   ↓
Grafana Visualization
```

---

# Sample Output

```text
Sound Level : 72.3 dB
RMS Value   : 0.621
State       : RUNNING
```

---

# Project Files

| File                  | Description              |
| --------------------- | ------------------------ |
| training_model.py     | CNN model training       |
| realtime_monitor.py   | Real-time monitoring     |
| feature_extraction.py | Audio feature extraction |
| adapter.py            | MTConnect adapter        |
| dashboard.sql         | Database setup           |
| README.md             | Project documentation    |

---

# Challenges

* Background industrial noise
* Overlapping RMS values
* Real-time processing latency
* Sensor placement sensitivity
* Audio data imbalance

---

# Future Improvements

* Improve dataset size
* Add anomaly detection
* Cloud integration
* Mobile dashboard support
* Predictive maintenance alerts
* Multi-machine monitoring
* Advanced deep learning models

---

# Results

The system successfully:

* Detects compressor operating conditions
* Performs real-time monitoring
* Stores industrial data automatically
* Visualizes live machine status
* Demonstrates Industrial IoT integration

---

# Authors

* Aarya Farheen
* Neha Venugopal

ME597 – Industrial Internet of Things (IIoT)
Purdue University

---

# License

This project is developed for educational and research purposes.
