import cv2
import face_recognition
import numpy as np
import sqlite3
import json
import requests
import time
import datetime
import os
import threading
import tkinter as tk
from PIL import Image, ImageTk
from dotenv import load_dotenv

load_dotenv()

# Backend API endpoints
API_PREFIX = os.getenv("API_PREFIX", "https://q9z46niut5.execute-api.ap-southeast-1.amazonaws.com")
API_FETCH_EMPLOYEES = API_PREFIX + "/employee"
API_UPLOAD_ACTIVITY = API_PREFIX + "/activity/sync"
MACHINE_ID = int(os.getenv("MACHINE_ID", 1))

# Global flag to control recognition state
recognition_active = True

# Load employee data from backend
def sync_employee_data():
    print("[SYNC] Starting to sync employee data from backend...")
    response = requests.get(API_FETCH_EMPLOYEES)
    if response.status_code == 200:
        employees = response.json()

        conn = sqlite3.connect("employees.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM employees")

        for emp in employees:
            cursor.execute("INSERT INTO employees (id, name, face_encoding) VALUES (?, ?, ?)",
                           (emp["id"], f"{emp['firstName']} {emp['lastName']}", emp["faceId"]))
        conn.commit()
        conn.close()
        print("[SYNC] Employee data synced successfully.")
    else:
        print(f"[SYNC] Failed to fetch employees. Status code: {response.status_code}")

# Load employee face encodings from local DB
def load_employee_data():
    print("[LOAD] Loading employee face data from local DB...")
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, face_encoding FROM employees")
    employees = cursor.fetchall()
    conn.close()

    known_face_encodings = []
    known_face_ids = []
    known_face_names = []

    for emp in employees:
        known_face_ids.append(emp[0])
        known_face_names.append(emp[1])
        known_face_encodings.append(np.array(json.loads(emp[2])))

    print("[LOAD] Employee face data loaded.")
    return known_face_ids, known_face_names, known_face_encodings

# Determine status (CHECK IN or CHECK OUT)
def get_status(emp_id):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE employee_id = ? AND DATE(timestamp) = ?", (emp_id, today))
    count = cursor.fetchone()[0]
    conn.close()
    return "CHECK IN" if count % 2 == 0 else "CHECK OUT"

# Record attendance locally
def record_attendance(emp_id, emp_name):
    print(f"[RECORD] Recording attendance for {emp_name}...")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = get_status(emp_id)

    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO attendance (employee_id, name, status, timestamp) VALUES (?, ?, ?, ?)",
                   (emp_id, emp_name, status, timestamp))
    conn.commit()
    conn.close()
    print(f"[RECORD] Attendance recorded for {emp_name} as {status}.")

# Upload and clean attendance
def upload_and_sync():
    global recognition_active
    print("[UPLOAD] Starting attendance upload and sync...")
    recognition_active = False

    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    cursor.execute("SELECT employee_id, name, status, timestamp FROM attendance")
    logs = cursor.fetchall()

    if logs:
        batch_size = 30
        for i in range(0, len(logs), batch_size):
            batch = logs[i:i + batch_size]
            activities = []
            for log in batch:
                activities.append({
                    "employee_id": log[0],
                    "status": log[2],
                    "created_at": log[3],
                    "machine_id": MACHINE_ID
                })
            try:
                response = requests.post(API_UPLOAD_ACTIVITY, json={"activities": activities})
                if response.status_code == 201:
                    print(f"[UPLOAD] Batch {i//batch_size + 1} uploaded successfully.")
                else:
                    print(f"[UPLOAD] Batch {i//batch_size + 1} failed. Status code: {response.status_code}")
            except Exception as e:
                print(f"[UPLOAD] Exception during upload: {e}")

        cursor.execute("CREATE TABLE IF NOT EXISTS attendance_bk AS SELECT * FROM attendance WHERE 0")
        cursor.execute("INSERT INTO attendance_bk SELECT * FROM attendance")
        cursor.execute("DELETE FROM attendance")
        conn.commit()

    conn.close()

    sync_employee_data()
    app.reload_face_data()
    recognition_active = True
    print("[UPLOAD] Attendance upload and sync completed.")
    threading.Timer(600, upload_and_sync).start()

# GUI Application
class FaceRecognitionApp:
    def __init__(self, root):
        print("[INIT] Initializing FaceRecognitionApp...")
        self.root = root
        self.root.title("Employee Face Recognition System")
        self.root.attributes("-fullscreen", True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)

        self.known_face_ids, self.known_face_names, self.known_face_encodings = load_employee_data()

        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height)
        self.canvas.pack()

        self.status_label = tk.Label(self.canvas, text="", font=("Helvetica", 24), bg="black", fg="white")
        self.status_label.place(x=20, y=20)

        self.last_detected = time.time()
        self.update_frame()

    def update_frame(self):
        if recognition_active:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.resize(frame, (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                recognized = False
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.45)
                    if True in matches:
                        match_index = matches.index(True)
                        emp_id = self.known_face_ids[match_index]
                        emp_name = self.known_face_names[match_index]

                        if time.time() - self.last_detected > 3:
                            self.last_detected = time.time()
                            record_attendance(emp_id, emp_name)
                            self.status_label.config(text=f"Recognized: {emp_name}")
                            self.last_recognized_time = time.time()
                        recognized = True

                # Clear label after 2 seconds
                if self.status_label.cget("text") and time.time() - self.last_recognized_time > 2:
                    self.status_label.config(text="")

                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk

        self.root.after(16, self.update_frame)

    def reload_face_data(self):
        print("[RELOAD] Reloading face data after sync...")
        self.known_face_ids, self.known_face_names, self.known_face_encodings = load_employee_data()
        print("[RELOAD] Reload completed.")

# Initialize local database
def setup_local_db():
    print("[DB] Setting up local database if not exists...")
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        face_encoding TEXT NOT NULL)""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        timestamp TEXT NOT NULL)""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS attendance_bk (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        timestamp TEXT NOT NULL)""")

    conn.commit()
    conn.close()
    print("[DB] Database setup completed.")

if __name__ == "__main__":
    print("[MAIN] Application starting...")
    setup_local_db()
    sync_employee_data()

    root = tk.Tk()
    app = FaceRecognitionApp(root)
    threading.Timer(600, upload_and_sync).start()
    print("[MAIN] Face recognition started.")
    root.mainloop()