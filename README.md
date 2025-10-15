# 🛡️ DrishtIQ — User & Entity Behavior Analytics System

> **AI-driven insider threat detection system** powered by machine learning and behavior analytics.  
> Built with **React (frontend)**, **Node.js (backend)**, and **Python (ML model)** to identify anomalous user activities using real log data.  
> Future scope includes **blockchain-based audit trails**, **desktop agent integration**, and **defense-grade tamper resistance**.

---

## 🚀 Overview

**DrishtIQ** is an intelligent platform that detects **insider threats** and **suspicious behavior** within an organization by analyzing user activity patterns.  

Traditional cybersecurity tools focus on *external attacks*, but insider threats often come from *trusted users*, making them harder to detect.  
DrishtIQ addresses this by using **machine learning** to learn normal behavioral patterns and flag deviations as potential threats.

---

## 🧩 Problem Statement

> **Problem ID:** 25244  
> **Organization:** Defence Research and Development Organisation (DRDO), Ministry of Defence  
> **Theme:** Blockchain & Cybersecurity  
> **Title:** User and Entity Behaviour Analytics (UEBA) for Internal Threat Identification

---

## 🔮 Full Scope (Planned for Full System)

- Real-time data ingestion using local endpoint agents  
- Blockchain integration for tamper-proof alert logging  
- Cross-platform Electron desktop app  
- Deep learning–based behavior analytics  
- Tamper detection and self-healing mechanisms  
- Integration with SIEM systems (Splunk/ELK)  
- Role-based access control (RBAC)  

---

## ⚙️ System Architecture (Current Prototype)

```
[User Activity Data / System Logs]
            ↓
[Data Processing & Normalization]
            ↓
[ML Model (Python)]
            ↓
[Anomaly Detection + Risk Scoring]
            ↓
[Node.js Backend API]
            ↓
[React Dashboard UI]
```

---

## 🧠 Core Components

### 1️⃣ Machine Learning Engine (Python)
- Reads **user behavior data** (from CSV logs).  
- Processes and normalizes features like login time, bytes sent, file size, and event frequency.  
- Uses **unsupervised ML algorithms** (e.g., Isolation Forest / One-Class SVM) for anomaly detection.  
- Generates:
  - `risk_scores.csv` — anomaly scores for each user/session  
  - `alerts.csv` — list of users/events exceeding anomaly threshold  
- Built using:  
  **Python**, **Pandas**, **Scikit-learn**, **NumPy**

---

### 2️⃣ Backend Server (Node.js)
- Acts as the **bridge** between ML and the frontend.  
- Responsible for:
  - Serving ML-generated data (alerts, scores, logs) via REST APIs.  
  - Handling client requests from React dashboard.  
  - Managing communication and updates between UI and ML engine.  
- Built using:  
  **Node.js**, **Express.js**, **Multer (for CSV uploads)**, **CORS**, **Axios**

---

### 3️⃣ Frontend Dashboard (React.js)
- Provides a **real-time visualization** of user and entity behavior.  
- Displays:
  - Risk scores
  - Anomaly alerts
  - Historical trends (charts)
  - CSV upload and analysis results  
- Designed with:  
  **React.js**, **Chart.js**, **Tailwind CSS**, **Axios**

---

### 4️⃣ Data Flow (Prototype)
1. Admin uploads or streams **user activity logs (CSV)**.  
2. Backend forwards data to ML engine for analysis.  
3. ML model computes **risk scores** and flags anomalies.  
4. Processed results are displayed in the React dashboard as visual charts and alert tables.

---

## 📊 Features (Current Prototype)

✅ Upload and process user activity logs (CSV)  
✅ ML-based anomaly detection and risk scoring  
✅ Interactive dashboard with alert visualization  
✅ REST APIs for ML-to-frontend communication  
✅ Real-time rendering of detection results  
✅ Modular, extensible architecture  

---

## 🧮 Machine Learning Logic (Simplified)

**Algorithm:** Isolation Forest  
**Steps:**
1. Train on normal behavior logs.  
2. Compute anomaly scores for each record.  
3. Label outliers as “suspicious.”  
4. Output a ranked list of alerts with confidence values.

---

## 🛡️ Security & Ethics Note

This project is built strictly for **ethical and research purposes**.  
It does **not** perform destructive actions or unauthorized monitoring.  
All data used for training and testing is simulated for demo purposes.

---

## 💬 Acknowledgements
- **Ministry of Defence, DRDO** – for the problem statement  
- **Smart India Hackathon 2025** – for providing the platform  
- **Open Source Libraries:** React, Node.js, Flask, Scikit-learn, Pandas  

---
