from fastapi import FastAPI, File, UploadFile, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import base64
from datetime import timedelta, datetime
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points (a, b, c) where b is the vertex."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))

def analyze_posture_from_frame(landmarks, posture_type: str = "squat"):
    """
    Analyzes posture based on MediaPipe landmarks and specified posture type.
    Returns a list of detected issues.
    """
    issues = []
    if not landmarks:
        return ["No pose detected"]

    lm = landmarks.landmark

    get_lm_coords = lambda name: [
        lm[mp_pose.PoseLandmark[name]].x,
        lm[mp_pose.PoseLandmark[name]].y,
        lm[mp_pose.PoseLandmark[name]].z
    ]

    if posture_type.lower() == "squat":
        try:
            left_shoulder = get_lm_coords("LEFT_SHOULDER")
            right_shoulder = get_lm_coords("RIGHT_SHOULDER")
            left_hip = get_lm_coords("LEFT_HIP")
            right_hip = get_lm_coords("RIGHT_HIP")
            left_knee = get_lm_coords("LEFT_KNEE")
            right_knee = get_lm_coords("RIGHT_KNEE")
            left_ankle = get_lm_coords("LEFT_ANKLE")
            right_ankle = get_lm_coords("RIGHT_ANKLE")
            left_foot_index = get_lm_coords("LEFT_FOOT_INDEX")
            right_foot_index = get_lm_coords("RIGHT_FOOT_INDEX")

            avg_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2, (left_shoulder[2] + right_shoulder[2]) / 2]
            avg_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2, (left_hip[2] + right_hip[2]) / 2]
            avg_knee = [(left_knee[0] + right_knee[0]) / 2, (left_knee[1] + right_knee[1]) / 2, (left_knee[2] + right_knee[2]) / 2]

            back_angle = calculate_angle(avg_shoulder, avg_hip, avg_knee)
            
            if back_angle < 150:
                issues.append("Back too hunched (Squat)")

            if left_knee[0] > left_foot_index[0] + 0.02: 
                issues.append("Left knee over toe (Squat)")
            if right_knee[0] < right_foot_index[0] - 0.02: 
                issues.append("Right knee over toe (Squat)")

        except KeyError:
            issues.append("Missing keypoints for squat analysis.")
        except Exception as e:
            issues.append(f"Squat analysis error: {str(e)}")

    elif posture_type.lower() == "desk_sitting":
        try:
            nose = get_lm_coords("NOSE")
            left_ear = get_lm_coords("LEFT_EAR")
            right_ear = get_lm_coords("RIGHT_EAR")
            left_shoulder = get_lm_coords("LEFT_SHOULDER")
            right_shoulder = get_lm_coords("RIGHT_SHOULDER")
            left_hip = get_lm_coords("LEFT_HIP")
            right_hip = get_lm_coords("RIGHT_HIP")
            left_eye = get_lm_coords("LEFT_EYE")
            right_eye = get_lm_coords("RIGHT_EYE")

            avg_ear = [(left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2, (left_ear[2] + right_ear[2]) / 2]
            avg_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2, (left_shoulder[2] + right_shoulder[2]) / 2]
            avg_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2, (left_hip[2] + right_hip[2]) / 2]
            avg_eye = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2, (left_eye[2] + right_eye[2]) / 2]

            neck_back_angle = calculate_angle(avg_ear, avg_shoulder, avg_hip)
            if neck_back_angle < 160: 
                issues.append("Hunched back or forward head (Desk)")

            if abs(avg_shoulder[0] - avg_hip[0]) > 0.05: 
                 issues.append("Leaning or slouching (Desk)")

            if abs(nose[1] - avg_eye[1]) > 0.03: 
                issues.append("Head tilted up or down (Desk)")

        except KeyError:
            issues.append("Missing keypoints for desk sitting analysis.")
        except Exception as e:
            issues.append(f"Desk analysis error: {str(e)}")
    else:
        issues.append("Unknown posture type specified.")

    return issues if issues else ["Good posture"]


@app.post('/analyze-video')
async def analyze_video(
    file: UploadFile = File(...),
    posture_type: str = Form("squat", description="Type of posture to analyze: 'squat' or 'desk_sitting'") 
):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_index = 0
        per_frame_feedback = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            frame_issues = []
            if results.pose_landmarks:
                frame_issues = analyze_posture_from_frame(results.pose_landmarks, posture_type)
            else:
                frame_issues = ["No pose detected"]

            per_frame_feedback.append({
                "frame": frame_index,
                "timestamp": format_timestamp(frame_index / fps),
                "issues": frame_issues
            })

            frame_index += 1

        cap.release()
        return {"message": "Video analysis complete!", "per_frame_feedback": per_frame_feedback} 
    except Exception as e:
        print(f"Error during video analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

class ImagePayload(BaseModel):
    image: str
    posture_type: str = "squat"

@app.post("/analyze-frame")
async def analyze_frame(payload: ImagePayload):
    try:
        header, encoded = payload.image.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(binary_data))
        
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        issues = []

        if results.pose_landmarks:
            issues = analyze_posture_from_frame(results.pose_landmarks, payload.posture_type)
        else:
            issues = ["No pose detected"]

        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "issues": issues
        }

    except Exception as e:
        print(f"Error during frame analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "FastAPI Posture Detection Backend is running!"}

