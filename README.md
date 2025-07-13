# 🧠 FastAPI Posture Detection Backend

This is a FastAPI backend that analyzes posture from videos or webcam frames using **MediaPipe**. It supports:

- 📹 Video file uploads
- 📷 Live webcam frame analysis

Designed to work with a React frontend (e.g. deployed via Vercel).

---

## 🔧 Features

- Frame-by-frame video analysis for bad posture.
- Real-time analysis of webcam images.
- Two posture modes:
  - `squat`
  - `desk_sitting`
- Returns structured feedback with timestamped issues.

---

## 🛠️ Tech Stack

- **FastAPI**
- **MediaPipe**
- **OpenCV**
- **PIL (Pillow)**
- **NumPy**

---

## 📦 Installation

```bash
git clone https://github.com/your-username/posture-backend.git
cd posture-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

▶️ Run the Server
uvicorn app:app --reload

By default, the backend runs at:
http://127.0.0.1:8000

🌐 Expose Backend to Internet (using ngrok)
To connect your local backend with a deployed frontend (e.g. on Vercel), use ngrok to get a public HTTPS URL.

1. Install ngrok
npm install -g ngrok  # or use the desktop app from ngrok.com

2. Start ngrok tunnel
ngrok http 8000

This will generate a public URL like:
https://69dbe62bb51e.ngrok-free.app

🔧 Configure Frontend to Use Backend
In your frontend project, create a .env file and add:

REACT_APP_BACKEND_VIDEO_URL=https://69dbe62bb51e.ngrok-free.app/analyze-video
REACT_APP_BACKEND_FRAME_URL=https://69dbe62bb51e.ngrok-free.app/analyze-frame

✅ Now your frontend deployed on Vercel can call the local backend securely!
