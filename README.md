# 🎓 ExamInvigilatorAI

### 🤖 AI-Powered Real-Time Exam Invigilation & Malpractice Detection System

ExamInvigilatorAI is a computer vision-based examination monitoring system that analyzes student behavior in video streams, detects potentially suspicious activities, and generates video evidence for review.

The system integrates **YOLO-based object detection, multi-object tracking, pose estimation, behavior analysis, rule-based event detection, and a React-based monitoring dashboard**.

---

## ✨ Key Features

- 🎥 **Real-Time Video Processing** — Processes video frames using OpenCV.
- 🧠 **AI-Based Detection** — Uses YOLO-based models for visual detection.
- 🎯 **Multi-Object Tracking** — Tracks detected people across frames.
- 🧍 **Pose & Orientation Analysis** — Analyzes head pose, body orientation, and movement.
- 🚨 **Violation Detection** — Evaluates behavioral observations using configurable rules.
- 📹 **Evidence Generation** — Produces MP4 evidence clips for registered events.
- 🖥️ **Live Monitoring Dashboard** — Provides a web interface for monitoring sessions.
- 📋 **Violation Review** — Allows users to inspect detected events.
- 🗂️ **Historical Evidence** — Supports access to previously recorded evidence clips.
- 💚 **Backend Health Monitoring** — Checks backend availability through API endpoints.

---

## 🛠️ Technology Stack

| 🔧 Component | 💻 Technologies |
|---|---|
| 🎨 Frontend | React, TypeScript, Vite |
| ⚙️ Backend | Python, FastAPI, Uvicorn |
| 👁️ Computer Vision | OpenCV, YOLO |
| 🎯 Object Tracking | Kalman Filter, Hungarian Matching, Tracking Manager |
| 🧍 Behavior Analysis | Head-Pose Estimation, Orientation Analysis, Behavior Fusion |
| 🚨 Event Processing | Rule Engine, Violation Service, Temporal History |
| 📹 Evidence Management | Video Buffering, Frame Selection, MP4 Generation |
| 🗄️ Data Storage | SQLite and filesystem-based evidence storage |

---

## 🏗️ System Architecture

The application consists of three major layers.

### 🎨 1. Frontend Layer

The frontend provides the user interface for interacting with the examination monitoring system.

**Main components:**

- 📄 `DemoWorkspace.tsx` — Main monitoring interface.
- 📄 `ExamInvigilationPage.tsx` — Examination monitoring page.
- 📄 `SiteChrome.tsx` — Navigation and application layout.
- 🔌 `lib/api.ts` — Communicates with backend API endpoints.
- 🩺 `useBackendStatus.ts` — Monitors backend health.
- 🎬 `useExamSamples.ts` — Retrieves sample videos.
- 📝 `types/exam.ts` — Defines shared TypeScript data types.

### ⚙️ 2. Backend API Layer

The backend uses FastAPI to provide endpoints for live sessions, video processing, violations, evidence clips, and health monitoring.

**Main modules:**

- 🐍 `app/main.py` — Backend application entry point.
- 🔴 `app/api/v1/live.py` — Live-session operations.
- 🚨 `app/api/v1/violations.py` — Violation-related operations.
- 📹 `app/api/v1/clips.py` — Evidence clip access.
- 🎥 `app/api/v1/video.py` — Video-related operations.
- 💚 `app/api/v1/health.py` — Backend health checks.

### 🧠 3. AI Processing Pipeline

The AI pipeline connects video capture, detection, tracking, behavior analysis, event evaluation, and evidence generation.

**Processing stages:**

1. 🎥 Capture video frames.
2. 🧠 Detect objects using YOLO.
3. 🎯 Track detected people across frames.
4. 🆔 Maintain tracking information and student identities.
5. 🧍 Estimate head pose and body orientation.
6. 🔍 Analyze and combine behavioral observations.
7. ⚖️ Evaluate behavior using the rule engine.
8. 🚨 Register detected violation events.
9. 📹 Select relevant frames and generate evidence clips.
10. 🔄 Update live-session information for the frontend.

---

## 🔄 How the System Works

```text
🎥 Video Input
      │
      ▼
🧠 YOLO Object Detection
      │
      ▼
🎯 Multi-Object Tracking
      │
      ▼
🧍 Pose & Behavior Analysis
      │
      ▼
⚖️ Rule Engine Evaluation
      │
      ▼
🚨 Violation Event Registration
      │
      ▼
📹 Evidence Clip Generation
      │
      ▼
🖥️ Live Dashboard & Review
```

### 📌 Workflow Explanation

- **🎥 Video Input:** The backend captures frames from the selected video source.
- **🧠 Detection:** The detection pipeline identifies relevant objects and people.
- **🎯 Tracking:** The tracking system maintains identities across frames.
- **🧍 Behavior Analysis:** Pose, orientation, and movement observations are combined.
- **⚖️ Rule Evaluation:** The rule engine checks observations against configured conditions.
- **🚨 Event Registration:** Potential violations are recorded by the violation service.
- **📹 Evidence Generation:** Relevant frames are selected to create evidence clips.
- **🖥️ Dashboard:** The frontend retrieves session status and violation information for review.

---

## 📁 Project Structure

```text
📦 ExamInvigilatorAI/
│
├── 📂 backend/
│   ├── 📂 app/
│   │   ├── 📂 api/
│   │   │   └── 📂 v1/
│   │   │       ├── 📄 live.py
│   │   │       ├── 📄 violations.py
│   │   │       ├── 📄 clips.py
│   │   │       ├── 📄 video.py
│   │   │       └── 📄 health.py
│   │   │
│   │   ├── 📂 behavior/
│   │   ├── 📂 camera/
│   │   ├── 📂 database/
│   │   ├── 📂 detection/
│   │   ├── 📂 tracking/
│   │   └── 📄 main.py
│   │
│   ├── 📂 clips/
│   ├── 📂 sample_clips/
│   └── 📄 requirements.txt
│
├── 📂 frontend/
│   ├── 📂 src/
│   │   ├── 📂 components/
│   │   ├── 📂 hooks/
│   │   ├── 📂 lib/
│   │   ├── 📂 pages/
│   │   └── 📂 types/
│   │
│   ├── 📄 package.json
│   └── 📄 vite.config.ts
│
├── 📂 calibration/
├── 📄 .gitignore
└── 📄 README.md
```

*Note: This is a representative structure. Some filenames and directories may differ from the current repository.*

---

## ⚠️ Important Notes

- 🧑‍⚖️ Detected events indicate potentially suspicious behavior and should be reviewed by a human before being treated as confirmed malpractice.
- 🎯 Detection performance depends on video quality, lighting, camera placement, model accuracy, and configuration.
- 📹 Evidence generation depends on the availability of relevant video frames and successful recording.
- 🔐 Student video and behavioral data should be stored and handled securely.

---

## ▶️ Commands to Run the Project

### 🖥️ Terminal 1 — Backend

```powershell
cd ExamInvigilatorAI
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 🌐 Terminal 2 — Frontend

```powershell
cd ExamInvigilatorAI
cd frontend
npm run dev
```

---

## 🚀 Project Goal

To build an intelligent examination monitoring system that combines computer vision, behavioral analysis, automated event detection, and evidence management to assist human invigilators in reviewing examination activity.

**Developed with 🧠 AI | 👁️ Computer Vision | ⚙️ FastAPI | 🎨 React**
