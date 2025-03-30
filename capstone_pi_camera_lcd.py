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
from picamera2 import Picamera2  # Import Picamera2
from libcamera import Transform  # Import Transform from libcamera

load_dotenv()

# Backend API endpoints
API_PREFIX = os.getenv("API_PREFIX", "https://q9z46niut5.execute-api.ap-southeast-1.amazonaws.com/api")
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

# Determine status (CHECK IN or CHECK OUT) based on last record from both attendance tables
def get_status(emp_id):
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    
    # Query both attendance tables to find the most recent status
    cursor.execute("""
        SELECT status FROM (
            SELECT status, timestamp FROM attendance WHERE employee_id = ?
            UNION ALL
            SELECT status, timestamp FROM attendance_bk WHERE employee_id = ?
        )
        ORDER BY timestamp DESC
        LIMIT 1
    """, (emp_id, emp_id))

    last_record = cursor.fetchone()
    conn.close()

    # If no previous record, default to CHECK IN
    if not last_record:
        return "CHECK IN"

    # Return the opposite of the last status
    return "CHECK OUT" if last_record[0] == "CHECK IN" else "CHECK IN"

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

        # Initialize PiCamera2 instead of OpenCV capture
        self.picam2 = Picamera2()
        
        # Configure camera without specifying format - let it use default
        # This matches the working version's approach
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480)},
            transform=Transform(hflip=True)  # Mirror image for more natural interaction
        )
        self.picam2.configure(config)
        self.picam2.start()
        print("[CAMERA] PiCamera2 initialized and started")

        self.known_face_ids, self.known_face_names, self.known_face_encodings = load_employee_data()

        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height)
        self.canvas.pack()

        self.status_label = tk.Label(self.canvas, text="", font=("Helvetica", 24), bg="black", fg="white")
        self.status_label.place(x=20, y=20)
        
        # Add exit button
        self.exit_button = tk.Button(
            self.root, 
            text="EXIT", 
            font=("Helvetica", 16, "bold"),
            bg="red", 
            fg="white",
            command=self.on_closing,
            width=10,
            height=2
        )
        self.exit_button.place(x=screen_width-120, y=screen_height-80)

        self.last_detected = time.time()
        self.last_recognized_time = time.time()
        self.update_frame()

    def update_frame(self):
        if recognition_active:
            # Capture frame from PiCamera2
            frame = self.picam2.capture_array()
            
            # Extract only the first 3 channels (RGB) - this is key for proper colors
            frame = frame[:, :, :3]
            
            # Resize the frame to fit screen
            frame = cv2.resize(frame, (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
            
            # Use this frame for recognition
            rgb_frame = frame.copy()
            
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

            # Convert to PIL image and display
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        self.root.after(16, self.update_frame)

    def reload_face_data(self):
        print("[RELOAD] Reloading face data after sync...")
        self.known_face_ids, self.known_face_names, self.known_face_encodings = load_employee_data()
        print("[RELOAD] Reload completed.")
        
    def on_closing(self):
        # Clean up camera resources
        print("[EXIT] Shutting down application...")
        self.picam2.stop()
        self.root.destroy()
        os._exit(0)  # Force exit to terminate all threads

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
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    threading.Timer(600, upload_and_sync).start()
    print("[MAIN] Face recognition started.")
    root.mainloop()
