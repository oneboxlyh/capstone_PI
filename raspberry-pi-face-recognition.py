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
from picamera.array import PiRGBArray
from picamera import PiCamera

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

        # Initialize the PiCamera
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)  # Lower resolution for better performance
        self.camera.framerate = 24
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
        
        # Allow camera to warm up
        time.sleep(0.1)

        self.known_face_ids, self.known_face_names, self.known_face_encodings = load_employee_data()

        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height)
        self.canvas.pack()

        self.status_label = tk.Label(self.canvas, text="", font=("Helvetica", 24), bg="black", fg="white")
        self.status_label.place(x=20, y=20)

        self.last_detected = time.time()
        self.last_recognized_time = time.time()
        self.current_frame = None
        
        # Start camera capture
        self.start_capture()

    def start_capture(self):
        # Start a separate thread for camera capture to avoid blocking the GUI
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start frame processing
        self.update_frame()
    
    def capture_frames(self):
        # Continuous capture from PiCamera
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            if not recognition_active:
                self.rawCapture.truncate(0)
                continue
                
            self.current_frame = frame.array
            self.rawCapture.truncate(0)
            
            # Brief sleep to reduce CPU usage
            time.sleep(0.01)

    def update_frame(self):
        if recognition_active and self.current_frame is not None:
            frame = self.current_frame
            # Resize for display while maintaining aspect ratio
            h, w = frame.shape[:2]
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            
            # Calculate scaling factor
            scale = min(screen_w / w, screen_h / h)
            dim = (int(w * scale), int(h * scale))
            
            frame_resized = cv2.resize(frame, dim)
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Process every 3rd frame to reduce CPU load
            if int(time.time() * 10) % 3 == 0:
                # Use smaller scale for face detection to improve performance
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                    if True in matches:
                        match_index = matches.index(True)
                        emp_id = self.known_face_ids[match_index]
                        emp_name = self.known_face_names[match_index]

                        if time.time() - self.last_detected > 3:
                            self.last_detected = time.time()
                            record_attendance(emp_id, emp_name)
                            self.status_label.config(text=f"Recognized: {emp_name}")
                            self.last_recognized_time = time.time()

            # Clear label after 2 seconds
            if self.status_label.cget("text") and time.time() - self.last_recognized_time > 2:
                self.status_label.config(text="")

            # Display the frame
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image((screen_w - dim[0]) // 2, (screen_h - dim[1]) // 2, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        # Schedule the next frame update
        self.root.after(33, self.update_frame)  # ~30 FPS

    def reload_face_data(self):
        print("[RELOAD] Reloading face data after sync...")
        self.known_face_ids, self.known_face_names, self.known_face_encodings = load_employee_data()
        print("[RELOAD] Reload completed.")
        
    def cleanup(self):
        # Release camera resources properly
        self.camera.close()

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
    
    # Handle proper cleanup on application close
    def on_closing():
        global recognition_active
        recognition_active = False
        app.cleanup()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    threading.Timer(600, upload_and_sync).start()
    print("[MAIN] Face recognition started.")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_closing()
