import sys
import os
import time
import tkinter.simpledialog as simpledialog
import shutil
import cv2
import numpy as np
import face_recognition
import customtkinter as ctk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from datetime import datetime, timedelta
import csv
import subprocess
import glob


# Use ultralytics for all YOLO models
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Error: ultralytics not available. Please install: pip install ultralytics")

# TensorFlow for Keras models
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Error: tensorflow not available. Please install: pip install tensorflow")



# Paths
EMPLOYEE_DB = 'employee_db.csv'
MASK_MODEL_PATH = 'models/mask_yolov5.pt'
ATTENDANCE_LOG = os.path.join(os.path.dirname(__file__), 'attendance.csv')

# Helper functions
def load_employee_db():
    if not os.path.exists(EMPLOYEE_DB):
        return []
    try:
        with open(EMPLOYEE_DB, 'r', newline='') as f:
            reader = csv.reader(f)
            db = []
            for row in reader:
                if len(row) == 2:  # Simple format: [name, encoding]
                    name = row[0]
                    encoding = np.array(eval(row[1]))
                    db.append((name, encoding))
                elif len(row) >= 8:  # Extended format: [id, name, gender, issue, expiry, last_att, photo, encoding]
                    emp_id = row[0]
                    name = row[1]
                    gender = row[2]
                    issue_date = row[3]
                    expiry_date = row[4]
                    last_attendance = row[5]
                    photo_path = row[6]
                    encoding = np.array(eval(row[7]))
                    db.append((emp_id, name, gender, issue_date, expiry_date, last_attendance, photo_path, encoding))
                elif len(row) >= 2:  # Fallback for other formats
                    name = row[0]
                    encoding = np.array(eval(row[1]))
                    db.append((name, encoding))
            return db
    except Exception as e:
        print(f"Error loading database: {e}")
        return []

def save_employee_db(db):
    try:
        with open(EMPLOYEE_DB, 'w', newline='') as f:
            writer = csv.writer(f)
            for record in db:
                if len(record) == 2:  # Simple format: (name, encoding)
                    name, encoding = record
                    writer.writerow([name, encoding.tolist()])
                elif len(record) >= 8:  # Extended format: (id, name, gender, issue, expiry, last_att, photo, encoding)
                    emp_id, name, gender, issue_date, expiry_date, last_attendance, photo_path, encoding = record[:8]
                    writer.writerow([emp_id, name, gender, issue_date, expiry_date, last_attendance, photo_path, encoding.tolist()])
                else:  # Fallback
                    # Convert to simple format if unknown structure
                    writer.writerow([str(record[0]), record[-1].tolist()])  # Assume last element is encoding
    except Exception as e:
        print(f"Error saving database: {e}")

def log_attendance(name):
    try:
        # Create attendance log file with headers if it doesn't exist
        if not os.path.exists(ATTENDANCE_LOG):
            with open(ATTENDANCE_LOG, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Timestamp'])
        with open(ATTENDANCE_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    except Exception as e:
        print(f"Error logging attendance: {e}")
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror('Attendance Logging Error', f'Could not log attendance: {e}')
        except Exception:
            pass

def get_face_encoding(img):
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"Error getting face encoding: {e}")
        return None


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Attendance System')
        self.root.geometry('1600x800')
        self.root.resizable(True, True)
        ctk.set_appearance_mode("System")  # or "Dark", "Light"
        ctk.set_default_color_theme("blue")
        self.db = load_employee_db()
        self.available_models = []
        self.model_dir = r'C:\\Users\\Ernest\\Desktop\\ai2\\actual\\models'
        self.selected_model = 'yolov8.pt'  # Set default model
        self.camera_source = 0  # Default to PC camera
            
        # Migrate old-format records to new format on startup
        migrated = False
        for i, rec in enumerate(self.db):
            if len(rec) == 2:  # Old format: (name, encoding)
                name = rec[0]
                encoding = rec[1]
                emp_id = f"R{i+1:03d}"
                gender = "Male"
                issue = datetime.now().strftime('%d/%m/%Y')
                expiry = (datetime.now() + timedelta(days=365)).strftime('%d/%m/%Y')
                last_att = ""
                photo = "default_male.png"
                self.db[i] = (emp_id, name, gender, issue, expiry, last_att, photo, encoding)
                migrated = True
        if migrated:
            save_employee_db(self.db)
        self.yolo_model = None
        self.yolo_imgsz = None
        self.main_menu()

    def main_menu(self):
        # Reset MiniShell references to avoid stale widget errors
        self._minishell_embedded = None
        self._minishell_visible = False
        
        for widget in self.root.winfo_children():
            widget.destroy()
        dark_blue = '#10132a'
        self.root.configure(bg=dark_blue)
        # Main menu container fills the window, dark blue background
        self.frame = ctk.CTkFrame(self.root, fg_color=dark_blue, corner_radius=0)
        self.frame.pack(fill='both', expand=True)
        # Title
        title = ctk.CTkLabel(self.frame, text='Attendance System', font=('Segoe UI', 26, 'bold'), text_color='#fff')
        title.pack(pady=(40, 30))
        # Button style
        button_style = {
            'font': ('Segoe UI', 15, 'bold'),
            'fg_color': '#2563eb',
            'hover_color': '#1d4ed8',
            'text_color': 'white',
            'corner_radius': 12,
            'height': 48,
            'width': 260
        }
        admin_btn = ctk.CTkButton(self.frame, text='Admin Login', command=self.admin_login, **button_style)
        admin_btn.pack(pady=(0, 18))
        emp_btn = ctk.CTkButton(self.frame, text='Employee Login', command=self.employee_login, fg_color='#059669', hover_color='#047857', **{k: v for k, v in button_style.items() if k not in ['fg_color', 'hover_color']})
        emp_btn.pack(pady=(0, 18))
        view_btn = ctk.CTkButton(self.frame, text='View Attendance', command=self.view_attendance, fg_color='#7c3aed', hover_color='#5b21b6', **{k: v for k, v in button_style.items() if k not in ['fg_color', 'hover_color']})
        view_btn.pack(pady=(0, 18))
    # Footer removed as requested

    def admin_login(self):
        # Full-screen password entry (not a popup)
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.configure(bg='#18192a')

        dark_blue = '#10132a'
        frame = ctk.CTkFrame(self.root, fg_color=dark_blue, corner_radius=0)
        frame.pack(fill='both', expand=True)

        title = ctk.CTkLabel(frame, text='Admin Login', font=('Segoe UI', 22, 'bold'), text_color='#fff', fg_color='#18192a')
        title.pack(pady=(32, 10))
        pw_entry = ctk.CTkEntry(frame, show='*', placeholder_text='Enter admin password', width=220, font=('Segoe UI', 14))
        pw_entry.pack(pady=(10, 8))
        error_lbl = ctk.CTkLabel(frame, text='', font=('Segoe UI', 12), text_color='#ef4444', fg_color='#18192a')
        error_lbl.pack(pady=(0, 8))

        def on_submit():
            pwd = pw_entry.get()
            if pwd == '123':
                self.admin_menu()
            else:
                error_lbl.configure(text='Wrong password!', text_color='#ef4444')

        submit_btn = ctk.CTkButton(frame, text='Login', command=on_submit, fg_color='#2563eb', hover_color='#1d4ed8', font=('Segoe UI', 14, 'bold'), width=120, corner_radius=8)
        submit_btn.pack(pady=(8, 0))
        back_btn = ctk.CTkButton(frame, text='Back', command=self.main_menu, fg_color='#aaa', hover_color='#888', font=('Segoe UI', 12, 'bold'), width=120, corner_radius=8)
        back_btn.pack(pady=(12, 0))
        pw_entry.bind('<Return>', lambda event: on_submit())
        pw_entry.focus_set()

    def admin_menu(self):

        for widget in self.root.winfo_children():
            widget.destroy()
        dark_blue = '#10132a'
        self.root.configure(bg=dark_blue)

        # Expand the admin dashboard container to fill the window, no border, dark blue background
        frame = ctk.CTkFrame(self.root, fg_color=dark_blue, corner_radius=0)
        frame.pack(fill='both', expand=True)

        # Modern title and welcome
        title = ctk.CTkLabel(frame, text='Admin Dashboard', font=('Segoe UI', 24, 'bold'), text_color='#2563eb')
        title.pack(pady=(32, 8))
        subtitle = ctk.CTkLabel(frame, text='Welcome, Admin! What would you like to do?', font=('Segoe UI', 15), text_color='#fff')
        subtitle.pack(pady=(0, 24))

        # Modern action buttons (container now dark blue, no border)
        btn_frame = ctk.CTkFrame(frame, fg_color=dark_blue)
        btn_frame.pack(pady=10)
        btn_style = {
            'font': ('Segoe UI', 15, 'bold'),
            'fg_color': '#2563eb',
            'hover_color': '#1d4ed8',
            'text_color': 'white',
            'corner_radius': 10,
            'height': 44,
            'width': 220
        }
        add_emp_btn = ctk.CTkButton(btn_frame, text='Add New Employee', command=self.add_employee, **btn_style)
        add_emp_btn.pack(pady=(0, 16))
        view_att_btn = ctk.CTkButton(btn_frame, text='View Attendance Log', command=self.view_attendance, fg_color='#7c3aed', hover_color='#5b21b6', **{k: v for k, v in btn_style.items() if k not in ['fg_color', 'hover_color']})
        view_att_btn.pack(pady=(0, 16))
        manage_db_btn = ctk.CTkButton(btn_frame, text='Manage Employee Database', command=self.manage_db, fg_color='#059669', hover_color='#047857', **{k: v for k, v in btn_style.items() if k not in ['fg_color', 'hover_color']})
        manage_db_btn.pack(pady=(0, 16))


        today = datetime.now().strftime('%Y-%m-%d')
        attendance_today = []
        if os.path.exists(ATTENDANCE_LOG):
            with open(ATTENDANCE_LOG, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2 and row[1].startswith(today):
                        attendance_today.append(row[0])
        unique_today = set(attendance_today)

        info_card = ctk.CTkFrame(frame, fg_color='#f1f5f9', corner_radius=12)
        info_card.pack(pady=(24, 0), padx=32, fill='x')
        emp_count = len(self.db)
        info_lbl = ctk.CTkLabel(info_card, text=f"Total Employees: {emp_count}", font=('Segoe UI', 13, 'bold'), text_color='#2563eb', fg_color='#f1f5f9')
        info_lbl.pack(pady=(12, 0))
        att_lbl = ctk.CTkLabel(info_card, text=f"Today's Attendance: {len(unique_today)}", font=('Segoe UI', 13, 'bold'), text_color='#059669', fg_color='#f1f5f9')
        att_lbl.pack(pady=(0, 12))


        # --- Recent Activity Log ---
        activity_card = ctk.CTkFrame(frame, fg_color='#f8fafc', corner_radius=12)
        activity_card.pack(pady=(16, 0), padx=32, fill='x')
        act_title = ctk.CTkLabel(activity_card, text='Recent Attendance Activity', font=('Segoe UI', 13, 'bold'), text_color='#222', fg_color='#f8fafc')
        act_title.pack(pady=(10, 0))

        # Clear button
        def clear_recent_activity():
            if os.path.exists(ATTENDANCE_LOG):
                with open(ATTENDANCE_LOG, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Name', 'Timestamp'])
            self.admin_menu()

        clear_btn = ctk.CTkButton(activity_card, text='Clear Recent Activity', command=clear_recent_activity, fg_color='#ef4444', hover_color='#b91c1c', font=('Segoe UI', 12, 'bold'), width=160, corner_radius=8)
        clear_btn.pack(pady=(0, 8), anchor='e', padx=18)

        activity_list = []
        if os.path.exists(ATTENDANCE_LOG):
            with open(ATTENDANCE_LOG, 'r', newline='') as f:
                reader = list(csv.reader(f))
                for row in reversed(reader[1:]):  # skip header, show latest first
                    if len(row) >= 2:
                        activity_list.append(f"{row[0]}  @  {row[1]}")
                    if len(activity_list) >= 5:
                        break
        if activity_list:
            for entry in activity_list:
                ctk.CTkLabel(activity_card, text=entry, font=('Segoe UI', 12), text_color='#444', fg_color='#f8fafc').pack(anchor='w', padx=18)
        else:
            ctk.CTkLabel(activity_card, text='No recent activity.', font=('Segoe UI', 12), text_color='#aaa', fg_color='#f8fafc').pack(anchor='w', padx=18)

        # Add Back button to return to login page
        ctk.CTkButton(frame, text='Back', command=self.main_menu, font=('Segoe UI', 12, 'bold'), fg_color='#aaa', hover_color='#888', width=120, corner_radius=8).pack(pady=18)


    def add_employee(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        dark_blue = '#10132a'
        self.root.configure(bg=dark_blue)

        # Main frame fills the window, dark blue background
        frame = ctk.CTkFrame(self.root, fg_color=dark_blue, corner_radius=0)
        frame.pack(fill='both', expand=True)

        title = ctk.CTkLabel(frame, text='Add New Employee', font=('Segoe UI', 22, 'bold'), text_color='#fff', fg_color=dark_blue)
        title.pack(pady=(32, 10))
        subtitle = ctk.CTkLabel(frame, text='Enter name and choose how to add employee:', font=('Segoe UI', 14), text_color='#b3b8d1', fg_color=dark_blue)
        subtitle.pack(pady=(0, 18))

        # Layout: left = controls, right = webcam preview
        content = ctk.CTkFrame(frame, fg_color=dark_blue)
        content.pack(fill='both', expand=True, padx=30, pady=10)

        left = ctk.CTkFrame(content, fg_color=dark_blue)
        left.pack(side='left', fill='y', padx=(0, 40), pady=10)
        right = ctk.CTkFrame(content, fg_color=dark_blue)
        right.pack(side='left', fill='both', expand=True, pady=10)

        name_label = ctk.CTkLabel(left, text='Employee Name:', font=('Segoe UI', 13), text_color='#fff', fg_color=dark_blue)
        name_label.pack(pady=(0, 6))

        name_var = ctk.StringVar()
        name_entry = ctk.CTkEntry(left, textvariable=name_var, font=('Segoe UI', 14), width=220)
        name_entry.pack(pady=(0, 8))
        # Gender selection
        gender_var = ctk.StringVar(value='Male')
        gender_label = ctk.CTkLabel(left, text='Gender:', font=('Segoe UI', 13), text_color='#fff', fg_color=dark_blue)
        gender_label.pack(pady=(0, 2))
        gender_menu = ctk.CTkOptionMenu(left, variable=gender_var, values=['Male', 'Female'], width=120)
        gender_menu.pack(pady=(0, 18))

        # Helper to enable/disable upload/camera buttons
        def on_name_change(*args):
            name = name_var.get().strip()
            state = "normal" if name else "disabled"
            cam_btn.configure(state=state)
            upload_btn.configure(state=state)
        name_var.trace_add('write', on_name_change)

        # Webcam preview area (expanded)
        webcam_label = ctk.CTkLabel(right, text='Webcam preview will appear here', font=('Segoe UI', 13), text_color='#b3b8d1', fg_color='#181c3a', width=640, height=480, corner_radius=16)
        webcam_label.pack(pady=(0, 10), padx=10, fill='both', expand=True)

        # Initialize variables for webcam capture
        cap = None
        captures = []
        encodings = []
        count = [0]
        running = [True]

        def show_webcam():
            from PIL import Image, ImageTk
            nonlocal cap, captures, encodings, count, running  # Make variables accessible to capture_photo
            cap = cv2.VideoCapture(self.camera_source)
            # Set camera resolution to match preview area
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                webcam_label.configure(text='Could not open webcam.', image=None)
                return
            
            # Enable capture button when camera starts
            capture_btn.configure(state="normal")
            
            # Reset capture variables
            captures.clear()
            encodings.clear()
            count[0] = 0
            running[0] = True

            def update_frame():
                if not running[0]:
                    if cap.isOpened():
                        cap.release()
                    if webcam_label.winfo_exists():
                        webcam_label.configure(text='Webcam stopped.', image=None)
                    return
                ret, frame = cap.read()
                if not ret:
                    if webcam_label.winfo_exists():
                        webcam_label.configure(text='Failed to read from webcam.', image=None)
                    if cap.isOpened():
                        cap.release()
                    return
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (1400, 800))
                cv2.putText(frame, f'Captures: {count[0]}/5', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                cv2.putText(frame, 'Press SPACE or Capture button, ESC to stop', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(img)
                if webcam_label.winfo_exists():
                    webcam_label.imgtk = imgtk
                    webcam_label.configure(image=imgtk, text='')
                    webcam_label.after(30, update_frame)
            update_frame()

            def on_key(event):
                if event.keysym == 'space' and count[0] < 5:
                    capture_photo()  # Use the same function as the button
                elif event.keysym == 'Escape':
                    running[0] = False
                    if cap and cap.isOpened():
                        cap.release()
                    capture_btn.configure(state="disabled")
                    webcam_label.configure(text='Webcam stopped.', image=None)

            self.root.bind('<Key>', on_key)
            name_entry.focus_set()


        cam_btn = ctk.CTkButton(left, text='Add via Camera (5 Captures)', font=('Segoe UI', 14, 'bold'), fg_color='#2563eb', hover_color='#1d4ed8', width=220, corner_radius=10, command=show_webcam, state="disabled")
        cam_btn.pack(pady=(0, 8))

        # Capture button - initially disabled
        def capture_photo():
            nonlocal cap, captures, encodings, count
            if cap is None or not cap.isOpened():
                webcam_label.configure(text='Please start camera first.')
                return
            if count[0] < 5:
                ret, frame = cap.read()
                if not ret:
                    return
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (640, 480))
                enc = get_face_encoding(frame)
                if enc is not None:
                    encodings.append(enc)
                    captures.append(frame.copy())
                    count[0] += 1
                    webcam_label.configure(text=f'Captured {count[0]}/5')
                    if count[0] == 5:
                        # Disable capture button when done
                        capture_btn.configure(state="disabled")
                        cam_btn.configure(state="disabled")
                        cap.release()
                        webcam_label.configure(text='Done capturing.', image=None)
                        # Save images and encoding
                        name = name_var.get().strip()
                        if not name:
                            webcam_label.configure(text='Please enter a name.')
                            return
                        img_dir = os.path.join('datasets', 'employee_images', name)
                        os.makedirs(img_dir, exist_ok=True)
                        for i, img in enumerate(captures):
                            img_path = os.path.join(img_dir, f'cam_{i+1}.jpg')
                            cv2.imwrite(img_path, img)
                        avg_encoding = np.mean(encodings, axis=0)
                        self.db.append((name, avg_encoding))
                        save_employee_db(self.db)
                        messagebox.showinfo('Success', f'Employee "{name}" added successfully.')
                        self.admin_menu()
                else:
                    webcam_label.configure(text='No face detected. Try again.')

        capture_btn = ctk.CTkButton(left, text='📸 Capture', font=('Segoe UI', 14, 'bold'), fg_color='#059669', hover_color='#047857', width=220, corner_radius=10, command=capture_photo, state="disabled")
        capture_btn.pack(pady=(0, 16))

        # --- Upload Dashboard ---
        upload_dashboard = ctk.CTkFrame(left, fg_color='#181c3a', corner_radius=12)
        upload_dashboard.pack(fill='x', pady=(0, 16))
        upload_title = ctk.CTkLabel(upload_dashboard, text='Upload Images (5-10)', font=('Segoe UI', 13, 'bold'), text_color='#fff', fg_color='#181c3a')
        upload_title.pack(pady=(8, 2))
        upload_info = ctk.CTkLabel(upload_dashboard, text='Drag & drop or use button below', font=('Segoe UI', 11), text_color='#b3b8d1', fg_color='#181c3a')
        upload_info.pack(pady=(0, 6))
        uploaded_count_var = ctk.StringVar(value='0')
        uploaded_count_label = ctk.CTkLabel(upload_dashboard, textvariable=uploaded_count_var, font=('Segoe UI', 12), text_color='#fff', fg_color='#181c3a')
        uploaded_count_label.pack(pady=(0, 6))
        uploaded_files = []

        def update_uploaded_count():
            uploaded_count_var.set(f"Uploaded: {len(uploaded_files)} / 10")

        def on_files_selected():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror('Error', 'Please enter a name first.')
                return
            files = filedialog.askopenfilenames(
                title='Select face images (5-10)',
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not files:
                return
            if len(files) + len(uploaded_files) > 10:
                messagebox.showerror('Error', 'You can upload a maximum of 10 images.')
                return
            uploaded_files.extend(files[:10-len(uploaded_files)])
            update_uploaded_count()


        upload_btn = ctk.CTkButton(upload_dashboard, text='Select Images', command=on_files_selected, font=('Segoe UI', 12, 'bold'), fg_color='#059669', hover_color='#047857', width=180, corner_radius=8, state="disabled")
        upload_btn.pack(pady=(0, 8))

        # --- Ensure Confirm & Save, Clear Uploaded Images, and Back buttons are always visible ---
        def confirm_upload():
            name = name_var.get().strip()
            gender = gender_var.get()
            if not name:
                messagebox.showerror('Error', 'Please enter a name first.')
                return
            if len(uploaded_files) < 5:
                messagebox.showerror('Error', 'Please upload at least 5 images.')
                return
            encodings = []
            img_dir = os.path.join('datasets', 'employee_images', name)
            os.makedirs(img_dir, exist_ok=True)
            photo_path = ''
            for i, path in enumerate(uploaded_files):
                img = cv2.imread(path)
                if img is None:
                    continue
                ext = os.path.splitext(path)[1].lower()
                dest_path = os.path.join(img_dir, f'upload_{i+1}{ext}')
                try:
                    shutil.copy2(path, dest_path)
                except Exception:
                    cv2.imwrite(os.path.join(img_dir, f'upload_{i+1}.jpg'), img)
                if i == 0:
                    photo_path = dest_path
                enc = get_face_encoding(img)
                if enc is not None:
                    encodings.append(enc)
            if len(encodings) < 5:
                messagebox.showerror('Error', f'Could only detect faces in {len(encodings)} images. Need at least 5.')
                return
            avg_encoding = np.mean(encodings, axis=0)
            # Generate ID, issue/expiry dates
            next_id = f"R{len(self.db)+1:03d}"
            issue_date = datetime.now().strftime('%d/%m/%Y')
            expiry_date = (datetime.now() + timedelta(days=365)).strftime('%d/%m/%Y')
            # Use default photo if none uploaded
            if not photo_path:
                photo_path = 'default_female.png' if gender == 'Female' else 'default_male.png'
            # Add to db: (id, name, gender, issue, expiry, last_attendance, photo, encoding)
            self.db.append((next_id, name, gender, issue_date, expiry_date, '', photo_path, avg_encoding))
            save_employee_db(self.db)
            messagebox.showinfo('Success', f'Employee "{name}" added successfully.')
            self.admin_menu()

        confirm_btn = ctk.CTkButton(upload_dashboard, text='Confirm & Save', command=confirm_upload, font=('Segoe UI', 12, 'bold'), fg_color='#2563eb', hover_color='#1d4ed8', width=180, corner_radius=8)
        confirm_btn.pack(pady=(0, 8))

        def clear_uploaded():
            uploaded_files.clear()
            update_uploaded_count()

        clear_btn = ctk.CTkButton(upload_dashboard, text='Clear Uploaded Images', command=clear_uploaded, font=('Segoe UI', 12), fg_color='#ef4444', hover_color='#b91c1c', width=180, corner_radius=8)
        clear_btn.pack(pady=(0, 8))

        ctk.CTkButton(left, text='Back', command=self.admin_menu, font=('Segoe UI', 12, 'bold'), fg_color='#aaa', hover_color='#888', width=120, corner_radius=8).pack(pady=18)

        # Confirm and Save
        def confirm_upload():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror('Error', 'Please enter a name first.')
                return
            if len(uploaded_files) < 5:
                messagebox.showerror('Error', 'Please upload at least 5 images.')
                return
            encodings = []
            img_dir = os.path.join('datasets', 'employee_images', name)
            os.makedirs(img_dir, exist_ok=True)
            for i, path in enumerate(uploaded_files):
                img = cv2.imread(path)
                if img is None:
                    continue
                ext = os.path.splitext(path)[1].lower()
                dest_path = os.path.join(img_dir, f'upload_{i+1}{ext}')
                try:
                    shutil.copy2(path, dest_path)
                except Exception:
                    cv2.imwrite(os.path.join(img_dir, f'upload_{i+1}.jpg'), img)
                enc = get_face_encoding(img)
                if enc is not None:
                    encodings.append(enc)
            if len(encodings) < 5:
                messagebox.showerror('Error', f'Could only detect faces in {len(encodings)} images. Need at least 5.')
                return
            avg_encoding = np.mean(encodings, axis=0)
            self.db.append((name, avg_encoding))
            save_employee_db(self.db)
            messagebox.showinfo('Success', f'Employee "{name}" added successfully.')
            self.admin_menu()
            confirm_btn = ctk.CTkButton(upload_dashboard, text='Confirm & Save', command=confirm_upload, font=('Segoe UI', 12, 'bold'), fg_color='#2563eb', hover_color='#1d4ed8', width=180, corner_radius=8)
            confirm_btn.pack(pady=(0, 8))

        # Clear Uploaded Images button
        def clear_uploaded():
            uploaded_files.clear()
            update_uploaded_count()
            clear_btn = ctk.CTkButton(upload_dashboard, text='Clear Uploaded Images', command=clear_uploaded, font=('Segoe UI', 12), fg_color='#ef4444', hover_color='#b91c1c', width=180, corner_radius=8)
            clear_btn.pack(pady=(0, 8))

            # Back button
            ctk.CTkButton(left, text='Back', command=self.admin_menu, font=('Segoe UI', 12, 'bold'), fg_color='#aaa', hover_color='#888', width=120, corner_radius=8).pack(pady=18)


    def add_employee_camera(self):
        import shutil
        name = simpledialog.askstring('Add Employee', 'Enter employee name:')
        if not name:
            return
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            messagebox.showerror('Error', 'Could not open webcam.')
            return
        encodings = []
        images = []
        count = 0
        messagebox.showinfo('Instructions', 'Position your face in the camera and press SPACE to capture (5 times). Press ESC to cancel.')
        while count < 5:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror('Error', 'Failed to read from webcam.')
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f'Captures: {count}/5', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Press SPACE to capture, ESC to cancel', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Add Employee - Webcam Capture', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space key
                enc = get_face_encoding(frame)
                if enc is not None:
                    encodings.append(enc)
                    images.append(frame.copy())
                    count += 1
                    print(f'Captured {count}/5')
                else:
                    messagebox.showwarning('Warning', 'No face detected. Please ensure your face is clearly visible.')
            elif key == 27:  # ESC key
                break
        cap.release()
        cv2.destroyAllWindows()
        if len(encodings) < 5:
            messagebox.showerror('Error', f'Only captured {len(encodings)} face encodings. Need 5.')
            return
        # Save images to datasets/employee_images/{name}/
        img_dir = os.path.join('datasets', 'employee_images', name)
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(images):
            img_path = os.path.join(img_dir, f'cam_{i+1}.jpg')
            cv2.imwrite(img_path, img)
        avg_encoding = np.mean(encodings, axis=0)
        self.db.append((name, avg_encoding))
        save_employee_db(self.db)
        messagebox.showinfo('Success', f'Employee "{name}" added successfully.')
        self.admin_menu()

    def add_employee_upload(self):
        import shutil
        name = simpledialog.askstring('Add Employee', 'Enter employee name:')
        if not name:
            return
        file_paths = filedialog.askopenfilenames(
            title='Select face images (4-5)',
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_paths or len(file_paths) < 4:
            messagebox.showerror('Error', 'Please select at least 4 images.')
            return
        encodings = []
        img_dir = os.path.join('datasets', 'employee_images', name)
        os.makedirs(img_dir, exist_ok=True)
        for i, path in enumerate(file_paths[:5]):
            img = cv2.imread(path)
            if img is None:
                continue
            # Save a copy of the uploaded image
            ext = os.path.splitext(path)[1].lower()
            dest_path = os.path.join(img_dir, f'upload_{i+1}{ext}')
            try:
                shutil.copy2(path, dest_path)
            except Exception:
                # fallback: save as jpg if copy fails
                cv2.imwrite(os.path.join(img_dir, f'upload_{i+1}.jpg'), img)
            enc = get_face_encoding(img)
            if enc is not None:
                encodings.append(enc)
        if len(encodings) < 4:
            messagebox.showerror('Error', f'Could only detect faces in {len(encodings)} images. Need at least 4.')
            return
        avg_encoding = np.mean(encodings, axis=0)
        self.db.append((name, avg_encoding))
        save_employee_db(self.db)
        messagebox.showinfo('Success', f'Employee "{name}" added successfully.')
        self.admin_menu()

    def manage_db(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.configure(bg='#f5f6fa')

        frame = ctk.CTkFrame(self.root, fg_color='#ffffff', corner_radius=18)
        frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.92, relheight=0.88)

        title = ctk.CTkLabel(frame, text='Manage Employee Database', font=('Segoe UI', 22, 'bold'), text_color='#2563eb')
        title.pack(pady=(32, 8))
        subtitle = ctk.CTkLabel(frame, text='Edit or delete employee records below.', font=('Segoe UI', 14), text_color='#222')
        subtitle.pack(pady=(0, 18))


        # Table header
        table_frame = ctk.CTkFrame(frame, fg_color='#f8fafc', corner_radius=12)
        table_frame.pack(pady=10, padx=24, fill='both', expand=True)
        header = ctk.CTkFrame(table_frame, fg_color='#f1f5f9')
        header.pack(fill='x', padx=8, pady=(8, 0))
        ctk.CTkLabel(header, text='Name', font=('Segoe UI', 13, 'bold'), width=180, anchor='w', text_color='#222', fg_color='#f1f5f9').pack(side='left', padx=(8,0))
        ctk.CTkLabel(header, text='View', font=('Segoe UI', 13, 'bold'), width=80, anchor='center', text_color='#222', fg_color='#f1f5f9').pack(side='left')
        ctk.CTkLabel(header, text='Edit', font=('Segoe UI', 13, 'bold'), width=80, anchor='center', text_color='#222', fg_color='#f1f5f9').pack(side='left')
        ctk.CTkLabel(header, text='Delete', font=('Segoe UI', 13, 'bold'), width=80, anchor='center', text_color='#222', fg_color='#f1f5f9').pack(side='left')

        # Table rows
        for idx, rec in enumerate(self.db):
            # Support both new and old format records
            if len(rec) == 8:
                emp_id, name, gender, issue, expiry, last_att, photo, encoding = rec
            elif len(rec) == 2:
                name, encoding = rec
                emp_id = gender = issue = expiry = last_att = photo = ''
            else:
                continue  # skip malformed records
            row = ctk.CTkFrame(table_frame, fg_color='#fff' if idx%2==0 else '#f3f4f6')
            row.pack(fill='x', padx=8, pady=2)
            ctk.CTkLabel(row, text=name, font=('Segoe UI', 12), width=180, anchor='w', text_color='#222', fg_color=row.cget('fg_color')).pack(side='left', padx=(8,0))

            def make_view_callback(emp_name):
                def view():
                    self.view_employee_images(emp_name)
                return view
            def make_edit_callback(emp_name):
                def edit():
                    self.edit_employee(emp_name)
                return edit
            def make_delete_callback(emp_name):
                def delete():
                    self.delete_employee(emp_name)
                return delete

            ctk.CTkButton(row, text='View', width=70, font=('Segoe UI', 12, 'bold'), fg_color='#7c3aed', hover_color='#5b21b6', command=make_view_callback(name)).pack(side='left', padx=8)
            ctk.CTkButton(row, text='Edit', width=70, font=('Segoe UI', 12, 'bold'), fg_color='#2563eb', hover_color='#1d4ed8', command=make_edit_callback(name)).pack(side='left', padx=8)
            ctk.CTkButton(row, text='Delete', width=70, font=('Segoe UI', 12, 'bold'), fg_color='#ef4444', hover_color='#b91c1c', command=make_delete_callback(name)).pack(side='left', padx=8)

        # Add a single Back button at the bottom of the manage_db screen (after the table, outside the loop)
        back_btn = ctk.CTkButton(frame, text='Back', command=self.admin_menu, font=('Segoe UI', 12, 'bold'), fg_color='#aaa', hover_color='#888', width=120, corner_radius=8)
        back_btn.pack(pady=18)
    
    def view_employee_images(self, emp_name):
        import glob
        from PIL import Image, ImageTk
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.configure(bg='#f5f6fa')

        frame = ctk.CTkFrame(self.root, fg_color='#ffffff', corner_radius=18)
        frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.8, relheight=0.8)

        title = ctk.CTkLabel(frame, text=f'Images for {emp_name}', font=('Segoe UI', 22, 'bold'), text_color='#2563eb')
        title.pack(pady=(32, 10))
        subtitle = ctk.CTkLabel(frame, text='Below are the images captured or uploaded for this employee.', font=('Segoe UI', 14), text_color='#222')
        subtitle.pack(pady=(0, 18))

        # Images are stored in datasets/employee_images/{emp_name}/*.jpg|png|jpeg|bmp
        img_dir = os.path.join('datasets', 'employee_images', emp_name)
        img_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        # Remove all Back buttons before adding a single one at the end
        if not img_paths:
            ctk.CTkLabel(frame, text='No images found for this employee.', font=('Segoe UI', 13), text_color='#ef4444').pack(pady=20)
        else:
            # Display images in a grid (max 4 per row)
            grid_frame = ctk.CTkFrame(frame, fg_color='#ffffff')
            grid_frame.pack(padx=24, pady=10, fill='both', expand=True)
            thumb_size = (140, 140)
            max_per_row = 4
            images = []  # Keep references to avoid garbage collection
            for i, img_path in enumerate(img_paths):
                try:
                    img = Image.open(img_path)
                    img.thumbnail(thumb_size)
                    imgtk = ImageTk.PhotoImage(img)
                    images.append(imgtk)
                    lbl = ctk.CTkLabel(grid_frame, image=imgtk, text='')
                    lbl.grid(row=i//max_per_row, column=i%max_per_row, padx=10, pady=10)
                except Exception as e:
                    ctk.CTkLabel(grid_frame, text=f'Error loading image: {os.path.basename(img_path)}', font=('Segoe UI', 11), text_color='#ef4444').grid(row=i//max_per_row, column=i%max_per_row, padx=10, pady=10)

        # Ensure only one Back button is created at the very end
        for widget in frame.winfo_children():
            if isinstance(widget, ctk.CTkButton) and widget.cget('text') == 'Back':
                widget.destroy()
        ctk.CTkButton(frame, text='Back', command=self.manage_db, font=('Segoe UI', 12, 'bold'), fg_color='#aaa', hover_color='#888', width=120, corner_radius=8).pack(pady=18)

    def edit_employee(self, emp_name):
        # Find the employee record - handle both data formats
        idx = None
        emp = None
        for i, rec in enumerate(self.db):
            # Handle both formats: (name, encoding) and (id, name, gender, issue, expiry, last_attendance, photo, encoding)
            if len(rec) == 2:  # Simple format: (name, encoding)
                if rec[0] == emp_name:
                    idx = i
                    emp = rec
                    break
            elif len(rec) >= 8:  # Extended format: (id, name, gender, issue, expiry, last_attendance, photo, encoding)
                if isinstance(rec[1], str) and rec[1] == emp_name:
                    idx = i
                    emp = rec
                    break
            # Also check if rec[0] is the name in extended format (in case of different ordering)
            elif len(rec) > 2 and isinstance(rec[0], str) and rec[0] == emp_name:
                idx = i
                emp = rec
                break
        
        if idx is None or emp is None:
            messagebox.showerror('Error', f'Employee {emp_name} not found.')
            self.manage_db()
            return
        
        # Clear the screen
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.configure(bg='#10132a')

        # Expand the frame, remove background (transparent), and use dark blue for all backgrounds
        frame = ctk.CTkFrame(self.root, fg_color='transparent', corner_radius=0)
        frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.92, relheight=0.88)

        title = ctk.CTkLabel(frame, text='Edit Employee', font=('Segoe UI', 22, 'bold'), text_color='#fff')
        title.pack(pady=(32, 10))
        subtitle = ctk.CTkLabel(frame, text=f'Editing: {emp_name}', font=('Segoe UI', 14), text_color='#b3b8d1')
        subtitle.pack(pady=(0, 18))

        # Handle both data formats
        if len(emp) == 2:  # Simple format: (name, encoding)
            # Convert to extended format with defaults
            emp_id = f"R{idx+1:03d}"
            emp_name_val = emp[0]
            emp_gender = "Male"
            emp_issue = datetime.now().strftime('%d/%m/%Y')
            emp_expiry = (datetime.now() + timedelta(days=365)).strftime('%d/%m/%Y')
            emp_last_att = ""
            emp_photo = "default_male.png"
            emp_encoding = emp[1]
        else:  # Extended format
            emp_id = emp[0] if len(emp) > 0 else f"R{idx+1:03d}"
            emp_name_val = emp[1] if len(emp) > 1 else emp_name
            emp_gender = emp[2] if len(emp) > 2 else "Male"
            emp_issue = emp[3] if len(emp) > 3 else datetime.now().strftime('%d/%m/%Y')
            emp_expiry = emp[4] if len(emp) > 4 else (datetime.now() + timedelta(days=365)).strftime('%d/%m/%Y')
            emp_last_att = emp[5] if len(emp) > 5 else ""
            emp_photo = emp[6] if len(emp) > 6 else "default_male.png"
            emp_encoding = emp[7] if len(emp) > 7 else emp[1]  # fallback to old format

        # ID (view only)
        id_label = ctk.CTkLabel(frame, text='Employee ID:', font=('Segoe UI', 13), text_color='#b3b8d1')
        id_label.pack(pady=(0, 2))
        id_var = ctk.StringVar(value=emp_id)
        id_entry = ctk.CTkEntry(frame, textvariable=id_var, font=('Segoe UI', 14), width=260, state='disabled',
                                text_color='#b3b8d1', fg_color='#181c3a')
        id_entry.pack(pady=(0, 10))

        # Name (editable)
        name_label = ctk.CTkLabel(frame, text='Employee Name:', font=('Segoe UI', 13), text_color='#b3b8d1')
        name_label.pack(pady=(0, 2))
        name_var = ctk.StringVar(value=emp_name_val)
        name_entry = ctk.CTkEntry(frame, textvariable=name_var, font=('Segoe UI', 14), width=260,
                                text_color='#fff', fg_color='#181c3a')
        name_entry.pack(pady=(0, 10))

        # Gender (editable)
        gender_label = ctk.CTkLabel(frame, text='Gender:', font=('Segoe UI', 13), text_color='#b3b8d1')
        gender_label.pack(pady=(0, 2))
        gender_var = ctk.StringVar(value=emp_gender)
        gender_menu = ctk.CTkOptionMenu(frame, variable=gender_var, values=['Male', 'Female'], width=120,
                                    text_color='#fff', fg_color='#181c3a')
        gender_menu.pack(pady=(0, 10))

        # Issue date (view only)
        issue_label = ctk.CTkLabel(frame, text='Issue Date:', font=('Segoe UI', 13), text_color='#b3b8d1')
        issue_label.pack(pady=(0, 2))
        issue_var = ctk.StringVar(value=emp_issue)
        issue_entry = ctk.CTkEntry(frame, textvariable=issue_var, font=('Segoe UI', 14), width=260, state='disabled',
                                text_color='#b3b8d1', fg_color='#181c3a')
        issue_entry.pack(pady=(0, 10))

        # Expiry date (view only)
        expiry_label = ctk.CTkLabel(frame, text='Expiry Date:', font=('Segoe UI', 13), text_color='#b3b8d1')
        expiry_label.pack(pady=(0, 2))
        expiry_var = ctk.StringVar(value=emp_expiry)
        expiry_entry = ctk.CTkEntry(frame, textvariable=expiry_var, font=('Segoe UI', 14), width=260, state='disabled',
                                text_color='#b3b8d1', fg_color='#181c3a')
        expiry_entry.pack(pady=(0, 10))

        # Last attendance (view only)
        last_att_label = ctk.CTkLabel(frame, text='Last Attendance:', font=('Segoe UI', 13), text_color='#b3b8d1')
        last_att_label.pack(pady=(0, 2))
        last_att_var = ctk.StringVar(value=emp_last_att)
        last_att_entry = ctk.CTkEntry(frame, textvariable=last_att_var, font=('Segoe UI', 14), width=260, state='disabled',
                                    text_color='#b3b8d1', fg_color='#181c3a')
        last_att_entry.pack(pady=(0, 10))

        # Photo (editable)
        photo_label = ctk.CTkLabel(frame, text='Photo:', font=('Segoe UI', 13), text_color='#b3b8d1')
        photo_label.pack(pady=(0, 2))
        photo_path_var = ctk.StringVar(value=emp_photo)

        def select_photo():
            path = filedialog.askopenfilename(title='Select Photo', filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp')])
            if path:
                photo_path_var.set(path)

        photo_frame = ctk.CTkFrame(frame, fg_color='transparent')
        photo_frame.pack(pady=(0, 10))
        photo_entry = ctk.CTkEntry(photo_frame, textvariable=photo_path_var, font=('Segoe UI', 12), width=180,
                                text_color='#fff', fg_color='#181c3a')
        photo_entry.pack(side='left', padx=(0, 8))
        photo_btn = ctk.CTkButton(photo_frame, text='Change', font=('Segoe UI', 12), fg_color='#2563eb', hover_color='#1d4ed8', width=80, command=select_photo)
        photo_btn.pack(side='left')

        def save_edit():
            new_name = name_var.get().strip()
            new_gender = gender_var.get()
            new_photo = photo_path_var.get()
            if not new_name:
                name_label.configure(text='Employee Name: (cannot be empty)', text_color='#ef4444')
                return
            
            # Check for duplicate name (except self)
            for i, rec2 in enumerate(self.db):
                if i != idx:
                    # Check name in both formats
                    check_name = None
                    if len(rec2) == 2:  # Simple format
                        check_name = rec2[0]
                    elif len(rec2) >= 2:  # Extended format
                        check_name = rec2[1] if isinstance(rec2[1], str) else rec2[0]
                    
                    if check_name == new_name:
                        name_label.configure(text='Employee Name: (name already exists)', text_color='#ef4444')
                        return
            
            # Update record - always save in extended format
            updated = (
                emp_id,           # ID
                new_name,         # Name
                new_gender,       # Gender
                emp_issue,        # Issue date
                emp_expiry,       # Expiry date
                emp_last_att,     # Last attendance
                new_photo,        # Photo path
                emp_encoding      # Face encoding
            )
            
            self.db[idx] = updated
            save_employee_db(self.db)
            messagebox.showinfo('Success', f'Employee {new_name} updated successfully!')
            self.manage_db()

        btn_frame = ctk.CTkFrame(frame, fg_color='transparent')
        btn_frame.pack(pady=18)
        ctk.CTkButton(btn_frame, text='Save', command=save_edit, font=('Segoe UI', 13, 'bold'), fg_color='#2563eb', hover_color='#1d4ed8', width=120, corner_radius=8).pack(side='left', padx=8)
        ctk.CTkButton(btn_frame, text='Cancel', command=self.manage_db, font=('Segoe UI', 13, 'bold'), fg_color='#aaa', hover_color='#888', width=120, corner_radius=8).pack(side='left', padx=8)
    
    
    def delete_employee(self, emp_name):
        if messagebox.askyesno('Delete Employee', f'Are you sure you want to delete {emp_name}?'):
            # Remove employee handling both data formats
            new_db = []
            for rec in self.db:
                keep_record = True
                
                if len(rec) == 2:  # Simple format: (name, encoding)
                    if rec[0] == emp_name:
                        keep_record = False
                elif len(rec) >= 8:  # Extended format: (id, name, gender, issue, expiry, last_attendance, photo, encoding)
                    if isinstance(rec[1], str) and rec[1] == emp_name:
                        keep_record = False
                elif len(rec) > 2:  # Other formats - check if first element is the name
                    if isinstance(rec[0], str) and rec[0] == emp_name:
                        keep_record = False
                
                if keep_record:
                    new_db.append(rec)
            
            # Check if employee was actually found and deleted
            if len(new_db) == len(self.db):
                messagebox.showerror('Error', f'Employee {emp_name} not found for deletion.')
            else:
                self.db = new_db
                save_employee_db(self.db)
                messagebox.showinfo('Deleted', f'{emp_name} has been deleted.')
                
                # Also delete employee images directory if it exists
                import shutil
                img_dir = os.path.join('datasets', 'employee_images', emp_name)
                if os.path.exists(img_dir):
                    try:
                        shutil.rmtree(img_dir)
                        print(f"Deleted image directory for {emp_name}")
                    except Exception as e:
                        print(f"Could not delete image directory: {e}")
            
            self.manage_db()

    def remove_employee(self):
        if not self.db:
            messagebox.showinfo('Info', 'No employees in the database.')
            return
        
        # Simple dropdown selection
        employee_names = []
        for rec in self.db:
            if len(rec) == 2:
                employee_names.append(rec[0])
            elif len(rec) >= 8:
                employee_names.append(rec[1])
        
        # Create selection window
        selection_window = ctk.CTkToplevel(self.root)
        selection_window.title('Remove Employee')
        selection_window.geometry('350x200')
        selection_window.grab_set()
        ctk.CTkLabel(selection_window, text='Select Employee to Remove:', font=('Arial', 12, 'bold'), text_color='#222').pack(pady=20)
        selected_name = ctk.StringVar(value=employee_names[0])
        dropdown = ctk.CTkOptionMenu(selection_window, variable=selected_name, values=employee_names, width=200, height=32, font=('Arial', 11))
        dropdown.pack(pady=10)
        def confirm_remove():
            name_to_remove = selected_name.get()
            if messagebox.askyesno('Confirm', f'Remove employee "{name_to_remove}"?'):
                self.db = [(name, enc) for name, enc in self.db if name != name_to_remove]
                save_employee_db(self.db)
                messagebox.showinfo('Success', f'Employee "{name_to_remove}" removed.')
                selection_window.destroy()
                self.admin_menu()
        btn_frame = ctk.CTkFrame(selection_window)
        btn_frame.pack(pady=20)
        ctk.CTkButton(btn_frame, text='Remove', fg_color='#e74c3c', text_color='white', command=confirm_remove, width=100, height=32).pack(side='left', padx=5)
        ctk.CTkButton(btn_frame, text='Cancel', fg_color='#6c757d', text_color='white', command=selection_window.destroy, width=100, height=32).pack(side='left', padx=5)

    def add_employee_upload(self):
        name = simpledialog.askstring('Add Employee', 'Enter employee name:')
        if not name:
            return
        files = filedialog.askopenfilenames(
            title='Select face images (at least 3)',
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if len(files) < 3:
            messagebox.showerror('Error', 'Please select at least 3 images.')
            return
        encodings = []
        failed_files = []
        for file_path in files:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    failed_files.append(file_path)
                    continue
                enc = get_face_encoding(img)
                if enc is not None:
                    encodings.append(enc)
                else:
                    failed_files.append(file_path)
            except Exception as e:
                failed_files.append(file_path)
                print(f"Error processing {file_path}: {e}")
        if len(encodings) < 3:
            messagebox.showerror('Error', f'Could only detect faces in {len(encodings)} images. Need 3.')
            return
        # Average the encodings
        avg_encoding = np.mean(encodings, axis=0)
        self.db.append((name, avg_encoding))
        save_employee_db(self.db)
        success_msg = f'Employee "{name}" added successfully with {len(encodings)} face encodings.'
        if failed_files:
            success_msg += f'\nNote: {len(failed_files)} files failed to process.'
        messagebox.showinfo('Success', success_msg)
        # Always return to admin_menu after adding
        self.admin_menu()

    def add_employee_webcam(self):
        name = simpledialog.askstring('Add Employee', 'Enter employee name:')
        if not name:
            return
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            messagebox.showerror('Error', 'Could not open webcam.')
            return
        encodings = []
        count = 0
        messagebox.showinfo('Instructions', 'Position your face in the camera and press SPACE to capture (3 times). Press ESC to cancel.')
        while count < 3:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror('Error', 'Failed to read from webcam.')
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f'Captures: {count}/3', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Press SPACE to capture, ESC to cancel', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Add Employee - Webcam Capture', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space key
                enc = get_face_encoding(frame)
                if enc is not None:
                    encodings.append(enc)
                    count += 1
                    print(f'Captured {count}/3')
                else:
                    messagebox.showwarning('Warning', 'No face detected. Please ensure your face is clearly visible.')
            elif key == 27:  # ESC key
                break
        cap.release()
        cv2.destroyAllWindows()
        if len(encodings) < 3:
            messagebox.showerror('Error', f'Only captured {len(encodings)} face encodings. Need 3.')
            return
        # Average the encodings
        avg_encoding = np.mean(encodings, axis=0)
        self.db.append((name, avg_encoding))
        save_employee_db(self.db)
        messagebox.showinfo('Success', f'Employee "{name}" added successfully.')
        # Always return to admin_menu after adding
        self.admin_menu()

    def employee_login(self):
        # Reset MiniShell references to avoid stale widget errors
        self._minishell_embedded = None
        self._minishell_visible = False
        
        for widget in self.root.winfo_children():
            widget.destroy()
        dark_blue = '#10132a'
        self.root.configure(bg=dark_blue)
        frame = ctk.CTkFrame(self.root, fg_color=dark_blue, corner_radius=0)
        frame.pack(fill='both', expand=True)
        title = ctk.CTkLabel(frame, text='Employee Login', font=('Arial', 20, 'bold'), text_color='#2563eb')
        title.pack(pady=(40, 10))
        # Empty preview box
        preview_box = ctk.CTkLabel(frame, text='', fg_color='#181c3a', width=640, height=480, corner_radius=16)
        preview_box.place(relx=0.5, rely=0.5, anchor='center')
        info_label = ctk.CTkLabel(frame, text='Press Start to begin face recognition', font=('Arial', 13), text_color='#b3b8d1', fg_color=dark_blue)
        info_label.pack(pady=5)
        # Buttons
        btn_frame = ctk.CTkFrame(frame, fg_color=dark_blue)
        btn_frame.pack(pady=(10, 30))
        start_btn = ctk.CTkButton(btn_frame, text='Start', font=('Arial', 15, 'bold'), fg_color='#2563eb', hover_color='#1d4ed8', width=120)
        start_btn.pack(side='left', padx=10)
        end_btn = ctk.CTkButton(btn_frame, text='End', font=('Arial', 15, 'bold'), fg_color='#ef4444', hover_color='#b91c1c', width=120)
        end_btn.pack(side='left', padx=10)
        back_btn = ctk.CTkButton(btn_frame, text='Back', font=('Arial', 15, 'bold'), fg_color='#64748b', hover_color='#334155', width=120, command=self.main_menu)
        back_btn.pack(side='left', padx=10)
        # Face match threshold
        FACE_MATCH_THRESHOLD = 0.45
        def start_camera():
            nonlocal preview_box
            preview_box.destroy()
            info_label.configure(text='Press ESC to cancel')
            # Camera preview label (always centered)
            video_label = ctk.CTkLabel(frame, text='Camera starting...', font=('Arial', 12), fg_color='#dff9fb', corner_radius=10, width=640, height=480)
            video_label.place(relx=0.5, rely=0.5, anchor='center')
            cap = cv2.VideoCapture(self.camera_source)
            def show_frame():
                ret, frame_cv = cap.read()
                if not ret:
                    cap.release()
                    messagebox.showerror('Error', 'Failed to read from webcam.')
                    self.main_menu()
                    return
                frame_cv = cv2.flip(frame_cv, 1)
                rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, faces)
                face_labels = []
                for enc in encodings:
                    distances = []
                    for rec in self.db:
                        if len(rec) == 2:
                            db_name, db_enc = rec
                        elif len(rec) >= 8:
                            db_name, db_enc = rec[1], rec[7]
                        else:
                            continue
                        try:
                            dist = face_recognition.face_distance([db_enc], enc)[0]
                            distances.append((dist, db_name))
                        except Exception as e:
                            print(f"Error comparing faces: {e}")
                            continue
                    if distances:
                        distances.sort()
                        best_distance, best_name = distances[0]
                        print(f"Best match: {best_name} (distance={best_distance:.3f})")
                        if best_distance < FACE_MATCH_THRESHOLD:
                            face_labels.append((best_name, best_distance, True))
                        else:
                            face_labels.append((f"Closest: {best_name}", best_distance, False))
                    else:
                        face_labels.append(("Unknown", None, False))
                for i, (face, label_info) in enumerate(zip(faces, face_labels)):
                    top, right, bottom, left = face
                    name, best_distance, recognized = label_info
                    color = (0, 255, 0) if recognized else (255, 0, 0)
                    cv2.rectangle(frame_cv, (left, top), (right, bottom), color, 2)
                    label_text = name if recognized else f"{name} ({best_distance:.2f})" if best_distance is not None else "Unknown"
                    cv2.putText(frame_cv, label_text, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                try:
                    from PIL import Image, ImageTk
                    img = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
                    try:
                        img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    except AttributeError:
                        img = img.resize((640, 480), Image.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk, text="")
                except Exception as e:
                    print(f"Error displaying frame: {e}")
                    video_label.configure(text=f"Camera error: {str(e)}")
                recognized_indices = [i for i, (_, _, rec) in enumerate(face_labels) if rec]
                if recognized_indices:
                    cap.release()
                    recognized_name = face_labels[recognized_indices[0]][0]
                    
                    # Store the current user's face encoding for security verification
                    recognized_index = recognized_indices[0]
                    if recognized_index < len(encodings):
                        self.current_user_encoding = encodings[recognized_index]
                        self.current_user_name = recognized_name
                        print(f"Stored encoding for security verification: {recognized_name}")
                    
                    # Hide title, Start, End, and move Back to top left
                    title.pack_forget()
                    btn_frame.pack_forget()
                    # Place Back button at top left
                    top_left_back = ctk.CTkButton(frame, text='Back', font=('Arial', 15, 'bold'), fg_color='#64748b', hover_color='#334155', width=120, command=self.main_menu)
                    top_left_back.place(x=40, y=40)
                    # Show info card on the right side of the camera preview
                    self._show_info_card_side_by_side(frame, video_label, recognized_name)
                else:
                    video_label.after(30, show_frame)
            def on_escape(event):
                cap.release()
                self.main_menu()
            def end_camera():
                cap.release()
                video_label.destroy()
                end_btn.configure(state='normal')
                info_label.configure(text='Press Start to begin face recognition')
                preview_box = ctk.CTkLabel(frame, text='', fg_color='#181c3a', width=640, height=480, corner_radius=16)
                preview_box.place(relx=0.5, rely=0.5, anchor='center')
                start_btn.configure(state='normal')
                back_btn.configure(state='normal')
            self.root.bind('<Escape>', on_escape)
            self.root.focus_set()
            end_btn.configure(command=end_camera)
            show_frame()
        start_btn.configure(command=start_camera)

    def _show_info_card_side_by_side(self, parent_frame, video_label, name):
        # Remove any previous info card
        if hasattr(self, '_info_card_frame') and self._info_card_frame is not None:
            self._info_card_frame.destroy()
        # Find employee record
        emp = None
        for rec in self.db:
            if len(rec) == 2 and rec[0] == name:
                emp = {'id': '', 'name': rec[0], 'gender': '', 'issue': '', 'expiry': '', 'last_att': '', 'photo': '', 'encoding': rec[1]}
                break
            elif len(rec) >= 8 and rec[1] == name:
                emp = {'id': rec[0], 'name': rec[1], 'gender': rec[2], 'issue': rec[3], 'expiry': rec[4], 'last_att': rec[5], 'photo': rec[6], 'encoding': rec[7]}
                break
        if emp is None:
            return
        # Create a frame docked to the far right, fixed position
        info_card = ctk.CTkFrame(parent_frame, fg_color='#181c3a', corner_radius=18, width=420, height=600)
        info_card.place(relx=1.0, rely=0.5, anchor='e', x=-60)
        self._info_card_frame = info_card
        # Company logo/name
        company_label = ctk.CTkLabel(info_card, text='TARUMT', font=('Segoe UI', 28, 'bold'), text_color='#2563eb')
        company_label.pack(pady=(32, 8))
        # Employee photo
        from PIL import Image, ImageTk
        photo_path = emp['photo']
        img = None
        # If photo_path is a directory, pick the first image file inside
        if photo_path and os.path.exists(photo_path):
            if os.path.isdir(photo_path):
                import glob
                found_imgs = []
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                    found_imgs.extend(glob.glob(os.path.join(photo_path, ext)))
                if found_imgs:
                    photo_path = found_imgs[0]
            try:
                img = Image.open(photo_path).resize((180, 180))
            except Exception:
                img = None
        if img is None:
            img_dir = os.path.join('datasets', 'employee_images')
            if emp['gender'] and emp['gender'].lower() == 'female':
                default_path = os.path.join(img_dir, 'default_female.jpg')
            else:
                default_path = os.path.join(img_dir, 'default_male.jpeg')
            try:
                img = Image.open(default_path).resize((180, 180))
            except Exception:
                img = Image.new('RGB', (180, 180), color='#222')
        photo_imgtk = ImageTk.PhotoImage(img)
        photo_label = ctk.CTkLabel(info_card, image=photo_imgtk, text='')
        photo_label.image = photo_imgtk
        photo_label.pack(pady=(0, 12))
        # Employee name and ID
        name_label = ctk.CTkLabel(info_card, text=emp['name'], font=('Segoe UI', 26, 'bold'), text_color='#fff')
        name_label.pack()
        id_label = ctk.CTkLabel(info_card, text=f"ID: {emp['id']}", font=('Segoe UI', 18), text_color='#b3b8d1')
        id_label.pack()
        # Issue/expiry dates, department/role, last attendance
        info_text = f"Issue: {emp['issue']}   Expiry: {emp['expiry']}"
        ctk.CTkLabel(info_card, text=info_text, font=('Segoe UI', 16), text_color='#b3b8d1').pack(pady=(8, 0))
        if emp.get('last_att'):
            ctk.CTkLabel(info_card, text=f"Last attendance: {emp['last_att']}", font=('Segoe UI', 15), text_color='#b3b8d1').pack()
        if emp.get('role'):
            ctk.CTkLabel(info_card, text=f"Role: {emp['role']}", font=('Segoe UI', 15), text_color='#b3b8d1').pack()
        # Animated welcome message
        welcome_label = ctk.CTkLabel(info_card, text=f"Welcome, {emp['name']}!", font=('Segoe UI', 22, 'bold'), text_color='#22d3ee')
        welcome_label.pack(pady=(18, 0))
        # Progress indicator
        progress_label = ctk.CTkLabel(info_card, text="Step 1 of 2: Identity Verified", font=('Segoe UI', 16, 'bold'), text_color='#22d3ee')
        progress_label.pack(pady=(12, 0))
        # Continue button at the bottom right of the parent_frame (right_frame)
        def proceed_to_mask():
            info_card.pack_forget()  # Hide info card but keep right_frame
            self._show_mask_instruction(emp, parent_frame)
        continue_btn = ctk.CTkButton(parent_frame, text='Continue', font=('Segoe UI', 17, 'bold'), fg_color='#2563eb', hover_color='#1d4ed8', width=220, corner_radius=12, command=proceed_to_mask)
        continue_btn.pack(side='bottom', anchor='se', pady=(32, 60), padx=(0, 60))
    def _after_face_recognized(self, name):
        self.root.unbind('<Escape>')
        # Find employee record (support both formats)
        emp = None
        for rec in self.db:
            if len(rec) == 2 and rec[0] == name:
                emp = {'id': '', 'name': rec[0], 'gender': '', 'issue': '', 'expiry': '', 'last_att': '', 'photo': 'default_male.png', 'encoding': rec[1]}
                break
            elif len(rec) >= 8 and rec[1] == name:
                emp = {'id': rec[0], 'name': rec[1], 'gender': rec[2], 'issue': rec[3], 'expiry': rec[4], 'last_att': rec[5], 'photo': rec[6], 'encoding': rec[7]}
                break
        if emp is None:
            messagebox.showerror('Error', f'Employee {name} not found.')
            self.main_menu()
            return

        # Show info card popup (full screen overlay)
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.configure(bg='#10132a')
        frame = ctk.CTkFrame(self.root, fg_color='#181c3a', corner_radius=18)
        frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.6, relheight=0.7)

        # Company logo/name (replace with your logo if available)
        company_label = ctk.CTkLabel(frame, text='TARUMT', font=('Segoe UI', 22, 'bold'), text_color='#2563eb')
        company_label.pack(pady=(24, 4))

        # Employee photo
        from PIL import Image, ImageTk
        photo_path = emp['photo']
        if not photo_path or not os.path.exists(photo_path):
            # Use default based on gender
            if emp['gender'].lower() == 'female':
                photo_path = 'default_female.png'
            else:
                photo_path = 'default_male.png'
        # Try to load the image (fallback to a blank if not found)
        try:
            if os.path.exists(photo_path):
                img = Image.open(photo_path).resize((120, 120))
            else:
                img = Image.new('RGB', (120, 120), color='#222')
        except Exception:
            img = Image.new('RGB', (120, 120), color='#222')
        photo_imgtk = ImageTk.PhotoImage(img)
        photo_label = ctk.CTkLabel(frame, image=photo_imgtk, text='')
        photo_label.image = photo_imgtk
        photo_label.pack(pady=(0, 8))

        # Employee name and ID
        name_label = ctk.CTkLabel(frame, text=emp['name'], font=('Segoe UI', 20, 'bold'), text_color='#fff')
        name_label.pack()
        id_label = ctk.CTkLabel(frame, text=f"ID: {emp['id']}", font=('Segoe UI', 14), text_color='#b3b8d1')
        id_label.pack()

        # Issue/expiry dates, department/role, last attendance
        info_text = f"Issue: {emp['issue']}   Expiry: {emp['expiry']}"
        ctk.CTkLabel(frame, text=info_text, font=('Segoe UI', 13), text_color='#b3b8d1').pack(pady=(4, 0))
        if emp.get('last_att'):
            ctk.CTkLabel(frame, text=f"Last attendance: {emp['last_att']}", font=('Segoe UI', 13), text_color='#b3b8d1').pack()
        # Department/role (if available)
        if emp.get('role'):
            ctk.CTkLabel(frame, text=f"Role: {emp['role']}", font=('Segoe UI', 13), text_color='#b3b8d1').pack()

        # Animated welcome message
        welcome_label = ctk.CTkLabel(frame, text=f"Welcome, {emp['name']}!", font=('Segoe UI', 18, 'bold'), text_color='#22d3ee')
        welcome_label.pack(pady=(12, 0))

        # Progress indicator
        progress_label = ctk.CTkLabel(frame, text="Step 1 of 2: Identity Verified", font=('Segoe UI', 13, 'bold'), text_color='#22d3ee')
        progress_label.pack(pady=(8, 0))

        # Continue button
        def proceed_to_mask():
            for widget in self.root.winfo_children():
                widget.destroy()
            self._show_mask_instruction(emp)
        continue_btn = ctk.CTkButton(frame, text='Continue', font=('Segoe UI', 15, 'bold'), fg_color='#2563eb', hover_color='#1d4ed8', width=180, corner_radius=10, command=proceed_to_mask)
        continue_btn.pack(pady=(24, 0))

    def _show_mask_instruction(self, emp, parent_frame=None):
        if parent_frame is not None:
            # Clear only the left/camera side of parent_frame, keep info card on right
            for widget in getattr(parent_frame, 'winfo_children', lambda:[])():
                try:
                    side = widget.pack_info().get('side', None)
                except Exception:
                    continue
                if side != 'top' and side != 'right':
                    widget.destroy()
            left_frame = ctk.CTkFrame(parent_frame, fg_color='#181c3a', corner_radius=18)
            left_frame.pack(side='left', fill='both', expand=True, padx=(40, 0), pady=(40, 0))
            ctk.CTkLabel(left_frame, text='Step 2 of 2: Mask Verification', font=('Segoe UI', 15, 'bold'), text_color='#22d3ee').pack(pady=(24, 8))
            ctk.CTkLabel(left_frame, text='Please wear your mask for attendance.', font=('Segoe UI', 16), text_color='#fff').pack(pady=(0, 8))
            try:
                from PIL import Image, ImageTk
                mask_icon = Image.open('mask_icon.png').resize((80, 80)) if os.path.exists('mask_icon.png') else Image.new('RGB', (80, 80), color='#444')
            except Exception:
                mask_icon = None
            if mask_icon is not None:
                mask_imgtk = ImageTk.PhotoImage(mask_icon)
                mask_label = ctk.CTkLabel(left_frame, image=mask_imgtk, text='')
                mask_label.image = mask_imgtk
                mask_label.pack(pady=(0, 8))
            else:
                ctk.CTkLabel(left_frame, text='[Mask Icon]', font=('Segoe UI', 18), text_color='#fff').pack(pady=(0, 8))
            preview_box = ctk.CTkLabel(left_frame, text='', fg_color='#222', width=240, height=180, corner_radius=12)
            preview_box.pack(pady=8)
            ctk.CTkLabel(left_frame, text='Progress: Mask Verification', font=('Segoe UI', 13), text_color='#b3b8d1').pack(pady=(8, 0))
            
            def start_mask():
                model_dir = self.model_dir
                model_path = os.path.join(model_dir, self.selected_model) if self.selected_model else os.path.join(model_dir, 'yolov8.pt')
                mask_detected = self.mask_detection()
                if mask_detected:
                    log_attendance(emp['name'])
                    
                
                #self.main_menu()
            
            # Fix UnboundLocalError: use left_frame if available, else frame
            btn_parent = left_frame if 'left_frame' in locals() else frame
            
            # Add Command prompt button at bottom left of parent_frame
            import tkinter as tk
            class MiniShell(tk.Frame):
                def __init__(self, master, app=None, prefix="MiniCMD> "):
                    super().__init__(master, bg="black")
                    self.app = app  
                    
                    self.prefix = prefix
                    self.authenticated = False
                    self.password = "456"
                    self.text = tk.Text(
                        self, height=8, width=2, bg="black", fg="white", insertbackground="white"
                    )
                    self.text.pack(fill="both", expand=True)
                    self.text.bind("<Return>", self.execute)
                    self.text.bind("<KeyPress>", self.protect_prefix)
                    self.insert_password_prompt()
                    self.commands = {
                        "help": self.help,
                        "echo": self.echo,
                        "clear": self.clear,
                        "model": self.model_cmd,
                        "status": self.status_cmd,
                        "camera": self.camera_cmd
                    }

                def camera_cmd(self, args):
                    if not args or args[0] != "switch":
                        return "Usage: camera switch [default|ip|<ip/url>]"
                    if len(args) < 2:
                        return "Usage: camera switch [default|ip|<ip/url>]"
                    target = args[1]
                    if self.app is not None:
                        if target == "default":
                            self.app.camera_source = 0
                            return "Camera source set to default PC camera."
                        elif target == "ip":
                            ip_url = "http://192.168.100.28:8080/video"
                            self.app.camera_source = ip_url
                            return f"Camera source set to {ip_url}"
                        else:
                            self.app.camera_source = target
                            return f"Camera source set to {target}"
                    return "App not available"
                def protect_prefix(self, event):
                    cursor = self.text.index("insert")
                    if cursor < self.lock_index:
                        self.text.mark_set("insert", "end-1c")
                    if event.keysym == "BackSpace":
                        if self.text.compare("insert", "<=", self.lock_index):
                            return "break"
                def execute(self, event):
                    line = self.text.get(self.lock_index, "end-1c").strip()
                    if not line:
                        self.text.mark_set("insert", "end")
                        return "break"
                    
                    # Handle password authentication
                    if not self.authenticated:
                        if line == self.password:
                            self.authenticated = True
                            self.text.insert("end", "\nAuthenticated! Welcome to MiniCMD.\n")
                            self.insert_prompt()
                        else:
                            self.text.insert("end", "\nIncorrect password. Try again.\n")
                            self.insert_password_prompt()
                        return "break"
                    
                    # Handle commands after authentication
                    cmd_parts = line.split()
                    cmd, args = cmd_parts[0], cmd_parts[1:]
                    output = self.commands.get(cmd, self.unknown)(args)
                    if output:
                        self.text.insert("end", f"\n{output}")
                    self.text.insert("end", "\n")
                    self.insert_prompt()
                    return "break"
                def insert_prompt(self):
                    if self.authenticated:
                        self.text.insert("end", self.prefix)
                    else:
                        self.text.insert("end", "Password: ")
                    self.text.see("end")
                    self.lock_index = self.text.index("end-1c")
                
                def insert_password_prompt(self):
                    self.text.insert("end", "Password: ")
                    self.text.see("end")
                    self.lock_index = self.text.index("end-1c")
                def help(self, args): return "Commands: help, echo, clear, status, model"
                def echo(self, args): return " ".join(args) if args else ""
                def clear(self, args):
                    self.text.delete("1.0", "end")
                    if self.authenticated:
                        self.insert_prompt()
                    else:
                        self.insert_password_prompt()
                    return ""
                def unknown(self, args): return "Unknown command"

                def status_cmd(self, args):
                    if self.app is not None:
                        current_model = getattr(self.app, 'selected_model', 'yolov8.pt')
                        model_type = "YOLO" if current_model.endswith('.pt') else "Keras" if current_model.endswith('.keras') else "Unknown"
                        return f"Current model: {current_model} ({model_type})"
                    return "App not available"

                def model_cmd(self, args):
                    if not args:
                        return "Usage: model list | model use [modelname] | model list box"
                    subcmd = args[0]
                    if subcmd == "list" and len(args) > 1 and args[1] == "box":
                        # Show model selection GUI
                        self.show_model_selection_gui()
                        return "Model selection window opened."
                    elif subcmd == "list":
                        model_dir = r"C:\\Users\\Ernest\\Desktop\\ai2\\actual\\models"
                        if not os.path.exists(model_dir):
                            return "Model directory not found."
                        # Include both YOLO (.pt) and Keras (.keras) models
                        models = [f for f in os.listdir(model_dir) if f.endswith(('.pt', '.keras'))]
                        if self.app is not None:
                            self.app.available_models = models
                        if not models:
                            return "No models found."
                        # Show model types
                        model_list = []
                        for model in models:
                            if model.endswith('.pt'):
                                model_list.append(f"{model} (YOLO)")
                            elif model.endswith('.keras'):
                                model_list.append(f"{model} (Keras)")
                        return "Available models:\n" + "\n".join(model_list)
                    elif subcmd == "use":
                        if len(args) < 2:
                            return "Usage: model use [modelname]"
                        modelname = args[1]
                        if self.app is not None and modelname in self.app.available_models:
                            self.app.selected_model = modelname
                            return f"Model set to: {modelname}"
                        else:
                            return f"Model '{modelname}' not found. Use 'model list' to see available models."
                    else:
                        return "Unknown model command. Use 'model list', 'model use [modelname]', or 'model list box'."

                def show_model_selection_gui(self):
                    """Show a GUI window for model selection"""
                    import tkinter as tk
                    from tkinter import ttk
                    
                    # Create the model selection window
                    model_window = tk.Toplevel(self.master)
                    model_window.title("Model Configuration")
                    model_window.geometry("350x350")
                    model_window.resizable(False, False)
                    model_window.configure(bg="#f0f0f0")
                    
                    # Make window modal
                    model_window.transient(self.master)
                    model_window.grab_set()
                    
                    # Title
                    title_label = tk.Label(model_window, text="Select AI Model", font=("Segoe UI", 14, "bold"), 
                                        bg="#f0f0f0", fg="#2563eb")
                    title_label.pack(pady=(20, 10))
                    
                    # Current model display
                    current_model = getattr(self.app, 'selected_model', 'yolov8.pt') if self.app else 'None'
                    current_label = tk.Label(model_window, text=f"Current: {current_model}", 
                                            font=("Segoe UI", 10), bg="#f0f0f0", fg="#666")
                    current_label.pack(pady=(0, 15))
                    
                    # Model list frame
                    list_frame = tk.Frame(model_window, bg="#f0f0f0")
                    list_frame.pack(pady=10, padx=20, fill="both", expand=True)
                    
                    # Scrollable listbox
                    listbox_frame = tk.Frame(list_frame, bg="#f0f0f0")
                    listbox_frame.pack(fill="both", expand=True)
                    
                    scrollbar = tk.Scrollbar(listbox_frame)
                    scrollbar.pack(side="right", fill="y")
                    
                    model_listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, 
                                            font=("Segoe UI", 10), height=10) 
                    model_listbox.pack(side="left", fill="both", expand=True)
                    scrollbar.config(command=model_listbox.yview)
                    
                    # Populate the listbox with models
                    model_dir = r"C:\\Users\\Ernest\\Desktop\\ai2\\actual\\models"
                    available_models = []
                    
                    if os.path.exists(model_dir):
                        models = [f for f in os.listdir(model_dir) if f.endswith(('.pt', '.keras'))]
                        for model in models:
                            model_type = "(YOLO)" if model.endswith('.pt') else "(Keras)"
                            display_text = f"{model} {model_type}"
                            model_listbox.insert(tk.END, display_text)
                            available_models.append(model)
                        
                        # Select current model if it exists in the list
                        for i, model in enumerate(available_models):
                            if model == current_model:
                                model_listbox.selection_set(i)
                                model_listbox.see(i)
                                break
                    else:
                        model_listbox.insert(tk.END, "Model directory not found")
                    
                    # Button frame - make sure it's visible
                    btn_frame = tk.Frame(model_window, bg="#f0f0f0", height=50)
                    btn_frame.pack(side="bottom", pady=(10, 20), fill="x")
                    btn_frame.pack_propagate(False)  # Prevent frame from shrinking
                    
                    def apply_selection():
                        selection = model_listbox.curselection()
                        if selection and available_models:
                            selected_model = available_models[selection[0]]
                            if self.app is not None:
                                self.app.selected_model = selected_model
                                self.app.available_models = available_models
                            model_window.destroy()
                            # Confirm in shell
                            self.text.insert("end", f"\nModel applied: {selected_model}")
                            self.text.insert("end", "\n")
                            self.insert_prompt()
                        else:
                            # Show error in a small popup
                            error_popup = tk.Toplevel(model_window)
                            error_popup.title("Error")
                            error_popup.geometry("200x100")
                            error_popup.configure(bg="#f0f0f0")
                            error_popup.transient(model_window)
                            error_popup.grab_set()
                            tk.Label(error_popup, text="Please select a model", 
                                    font=("Segoe UI", 10), bg="#f0f0f0", fg="#ef4444").pack(pady=20)
                            tk.Button(error_popup, text="OK", command=error_popup.destroy,
                                    bg="#ef4444", fg="white", font=("Segoe UI", 9, "bold")).pack()

                    def cancel_selection():
                        model_window.destroy()

                    # Create a container frame for buttons to ensure they're centered
                    button_container = tk.Frame(btn_frame, bg="#f0f0f0")
                    button_container.pack(expand=True)

                    # Buttons
                    apply_btn = tk.Button(button_container, text="Apply", command=apply_selection,
                                        bg="#2563eb", fg="white", font=("Segoe UI", 10, "bold"),
                                        width=10, relief="flat", cursor="hand2", padx=10, pady=5)
                    apply_btn.pack(side="left", padx=(0, 10))

                    cancel_btn = tk.Button(button_container, text="Cancel", command=cancel_selection,
                                        bg="#6c757d", fg="white", font=("Segoe UI", 10, "bold"),
                                        width=10, relief="flat", cursor="hand2", padx=10, pady=5)
                    cancel_btn.pack(side="left")
            # Embed MiniShell directly in the left_frame, toggled by the button
            # Reset MiniShell references to avoid stale widget errors
            self._minishell_embedded = None
            self._minishell_visible = False
            
            def toggle_shell():
                if self._minishell_embedded is None:
                    self._minishell_embedded = MiniShell(left_frame, app=self)
                    self._minishell_embedded.pack(pady=(16, 0), fill="x")
                    self._minishell_visible = True
                else:
                    if self._minishell_visible:
                        self._minishell_embedded.pack_forget()
                        self._minishell_visible = False
                    else:
                        self._minishell_embedded.pack(pady=(16, 0), fill="x")
                        self._minishell_visible = True
            cmd_btn = ctk.CTkButton(left_frame, text='Command prompt', font=('Segoe UI', 12, 'bold'), 
                                fg_color='#404040', hover_color='#303030', width=140, corner_radius=8, 
                                command=toggle_shell)
            cmd_btn.pack(pady=(18, 0), anchor='w')
            
        else:
            # ...existing code for full-screen/standalone mask instruction...
            # Step 2: Mask instructions and live preview
            for widget in self.root.winfo_children():
                widget.destroy()
            self.root.configure(bg='#10132a')
            frame = ctk.CTkFrame(self.root, fg_color='#181c3a', corner_radius=18)
            frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.6, relheight=0.7)
            ctk.CTkLabel(frame, text='Step 2 of 2: Mask Verification', font=('Segoe UI', 15, 'bold'), text_color='#22d3ee').pack(pady=(24, 8))
            ctk.CTkLabel(frame, text='Please wear your mask for attendance.', font=('Segoe UI', 16), text_color='#fff').pack(pady=(0, 8))
            try:
                from PIL import Image, ImageTk
                mask_icon = Image.open('mask_icon.png').resize((80, 80)) if os.path.exists('mask_icon.png') else Image.new('RGB', (80, 80), color='#444')
            except Exception:
                mask_icon = None
            if mask_icon is not None:
                mask_imgtk = ImageTk.PhotoImage(mask_icon)
                mask_label = ctk.CTkLabel(frame, image=mask_imgtk, text='')
                mask_label.image = mask_imgtk
                mask_label.pack(pady=(0, 8))
            else:
                ctk.CTkLabel(frame, text='[Mask Icon]', font=('Segoe UI', 18), text_color='#fff').pack(pady=(0, 8))
            preview_box = ctk.CTkLabel(frame, text='', fg_color='#222', width=240, height=180, corner_radius=12)
            preview_box.pack(pady=8)
            ctk.CTkLabel(frame, text='Progress: Mask Verification', font=('Segoe UI', 13), text_color='#b3b8d1').pack(pady=(8, 0))
        
        def start_mask():
            model_dir = self.model_dir
            model_path = os.path.join(model_dir, self.selected_model) if self.selected_model else os.path.join(model_dir, 'yolov8.pt')
            mask_detected = self.mask_detection()
            if mask_detected:
                log_attendance(emp['name'])
                
        
        # Use the correct parent for the button
        btn_parent = left_frame if parent_frame is not None else frame
        start_btn = ctk.CTkButton(btn_parent, text='Start Mask Detection', font=('Segoe UI', 14, 'bold'), fg_color='#059669', hover_color='#047857', width=200, corner_radius=10, command=start_mask)
        start_btn.pack(pady=(18, 0))
            
        # Add Command prompt button at bottom left of root window
        def dummy_command():
            pass  # Does nothing as requested
        
        cmd_btn = ctk.CTkButton(self.root, text='Command prompt', font=('Segoe UI', 12, 'bold'), 
                               fg_color='#404040', hover_color='#303030', width=140, corner_radius=8, 
                               command=dummy_command)
        cmd_btn.place(x=40, y=self.root.winfo_height() - 80)  # Bottom left positioning



        
    def mask_detection(self, name=None):
        """Mask detection with persistent security: waits for original person, shows info card after attendance."""
        
        for widget in self.root.winfo_children():
            widget.destroy()
        dark_blue = '#10132a'
        self.root.configure(bg=dark_blue)
        frame = ctk.CTkFrame(self.root, fg_color=dark_blue, corner_radius=0)
        frame.pack(fill='both', expand=True)
        ctk.CTkLabel(frame, text='Face Mask Detection App 😷', font=('Segoe UI', 20, 'bold'), text_color='#22d3ee').pack(pady=(40, 16))
        preview_box = ctk.CTkLabel(frame, text='Loading model...', fg_color='#222', width=640, height=480, corner_radius=12)
        preview_box.pack(pady=16)
        model_dir = getattr(self, 'model_dir', r'C:\\Users\\Ernest\\Desktop\\ai2\\actual\\models')
        model_name = self.selected_model if hasattr(self, 'selected_model') and self.selected_model else 'yolov8.pt'
        model_path = os.path.join(model_dir, model_name)
        ctk.CTkLabel(frame, text=f'Model: {model_name}', font=('Segoe UI', 13), text_color='#b3b8d1').pack(pady=(8, 0))
        status_label = ctk.CTkLabel(frame, text='Starting...', font=('Segoe UI', 14), text_color='#fff')
        status_label.pack(pady=(10, 0))
        btn_frame = ctk.CTkFrame(frame, fg_color=dark_blue)
        btn_frame.pack(pady=(20, 0))
        run_detection = [True]
        mask_confirmed = [False]
        authorized_person_encoding = None
        verification_timer = [0]
        last_verification_time = [0]
        person_verified = [False]
        persistent_unauth = [False]
        persistent_auth = [False]
        auth_timer = [0]
        auth_start_time = [0]
        # Add timer variables for no person detection
        no_person_timer = [0]
        no_person_start_time = [0]
        # Get authorized person encoding and name
        emp_name = name if name else getattr(self, 'current_user_name', None)
        if hasattr(self, 'current_user_encoding') and self.current_user_encoding is not None:
            authorized_person_encoding = self.current_user_encoding
        # Add a flag to skip identity check after first recognition
        identity_verified = [False]
        # Camera
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            preview_box.configure(text='Camera error')
            return False
        def stop_detection():
            run_detection[0] = False
            if cap.isOpened():
                cap.release()
            status_label.configure(text='Detection stopped', text_color='#ef4444')
            return mask_confirmed[0]
        def back_and_stop():
            stop_detection()
            self.main_menu()

        back_btn = ctk.CTkButton(btn_frame, text='Back', command=back_and_stop, fg_color='#6c757d', width=120)
        back_btn.pack(padx=10)

        def refresh_detection():
            stop_detection()
            # Re-run mask_detection with the same name (emp_name)
            self.mask_detection(name=emp_name)

        refresh_btn = ctk.CTkButton(btn_frame, text='Refresh', command=refresh_detection, fg_color='#38bdf8', hover_color='#0ea5e9', width=120)
        refresh_btn.pack(padx=10, pady=(8, 0))
        # Load model
        model = None
        model_type = None
        if model_path.endswith('.keras'):
            if not TENSORFLOW_AVAILABLE:
                preview_box.configure(text='Please install tensorflow: pip install tensorflow')
                return False
            try:
                model = tf.keras.models.load_model(model_path)
                model_type = 'keras'
                status_label.configure(text=f'Keras model loaded: {model_name}', text_color='#22d3ee')
            except Exception as e:
                preview_box.configure(text=f'Keras model error: {str(e)}')
                return False
        else:
            if not ULTRALYTICS_AVAILABLE:
                preview_box.configure(text='Please install ultralytics: pip install ultralytics')
                return False
            try:
                model = YOLO(model_path)
                model_type = 'yolo'
                status_label.configure(text=f'YOLO model loaded: {model_name}', text_color='#22d3ee')
            except Exception as e:
                preview_box.configure(text=f'YOLO model error: {str(e)}')
                return False
        def detection_loop():
            if not run_detection[0]:
                return
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                try:
                    annotated_frame = frame.copy()
                    mask_detected_in_frame = False
                    current_time = time.time()
                    # Step 1: Verify if the authorized person is present (only if not already verified)
                    if not identity_verified[0] and authorized_person_encoding is not None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        # Check for multiple persons first
                        if len(face_locations) > 1:
                            status_label.configure(text='⚠️ ERROR: MORE THAN 1 PERSON DETECTED! Only one person allowed.', text_color='#ef4444')
                            verification_timer[0] = 0
                            auth_timer[0] = 0
                            no_person_timer[0] = 0  # Reset no person timer
                            persistent_unauth[0] = True
                            persistent_auth[0] = False
                            cv2.putText(annotated_frame, "ERROR: MORE THAN 1 PERSON DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        elif len(face_locations) == 0:
                            # Start or continue no person timer
                            if no_person_timer[0] == 0:
                                no_person_start_time[0] = current_time
                                no_person_timer[0] = 1
                                status_label.configure(text='⚠️ No person detected. Please position yourself in front of camera.', text_color='#ef4444')
                            else:
                                elapsed = current_time - no_person_start_time[0]
                                remaining = 20.0 - elapsed
                                if remaining > 0:
                                    status_label.configure(text=f'⚠️ No person detected. Please position yourself in front of camera. {remaining:.1f}s remaining', text_color='#ef4444')
                                else:
                                    status_label.configure(text='⚠️ TIMEOUT: No person detected for 20 seconds. Returning to main menu.', text_color='#ef4444')
                                    run_detection[0] = False
                                    cap.release()
                                    self.main_menu()
                                    return
                            verification_timer[0] = 0
                            auth_timer[0] = 0
                            persistent_unauth[0] = True
                            persistent_auth[0] = False
                        else:
                            # Reset no person timer when person is detected
                            no_person_timer[0] = 0
                            # Exactly one person detected, proceed with authorization check
                            person_verified[0] = False
                            for face_encoding in face_encodings:
                                matches = face_recognition.compare_faces([authorized_person_encoding], face_encoding, tolerance=0.6)
                                if matches[0]:
                                    person_verified[0] = True
                                    break
                            if not person_verified[0]:
                                status_label.configure(text='⚠️ SECURITY ALERT: Unauthorized person detected! Please return original person.', text_color='#ef4444')
                                verification_timer[0] = 0
                                auth_timer[0] = 0
                                persistent_unauth[0] = True
                                persistent_auth[0] = False
                                cv2.putText(annotated_frame, "UNAUTHORIZED PERSON DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            else:
                                if persistent_unauth[0]:
                                    if auth_timer[0] == 0:
                                        auth_start_time[0] = current_time
                                        auth_timer[0] = 1
                                        status_label.configure(text='✓ Original person detected! Stay for 2 seconds...', text_color='#22d3ee')
                                    else:
                                        elapsed = current_time - auth_start_time[0]
                                        remaining = 2.0 - elapsed
                                        if remaining > 0:
                                            status_label.configure(text=f'✓ Stay... {remaining:.1f}s', text_color='#22d3ee')
                                        else:
                                            persistent_unauth[0] = False
                                            persistent_auth[0] = True
                                            auth_timer[0] = 0
                                else:
                                    persistent_auth[0] = True
                                # Mark identity as verified after successful check
                                identity_verified[0] = True
                    else:
                        # Already verified, skip identity check but still check for multiple people
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        
                        # Check for multiple persons even when identity is verified
                        if len(face_locations) > 1:
                            status_label.configure(text='⚠️ ERROR: MORE THAN 1 PERSON DETECTED! Only one person allowed.', text_color='#ef4444')
                            verification_timer[0] = 0
                            persistent_auth[0] = False
                            cv2.putText(annotated_frame, "ERROR: MORE THAN 1 PERSON DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        else:
                            # Authorized person count is acceptable  
                            persistent_auth[0] = True
                            person_verified[0] = True
                            identity_verified[0] = True
                    
                    # Step 2: Mask Detection (only if authorized person is present or already verified)
                    if (person_verified[0] or authorized_person_encoding is None) and persistent_auth[0]:
                        name_to_show = emp_name if emp_name else "Unknown"
                        if model_type == 'yolo':
                            results = model(frame, conf=0.3, iou=0.45)
                            if results and len(results) > 0:
                                result = results[0]
                                if hasattr(result, 'boxes') and result.boxes is not None:
                                    boxes = result.boxes
                                    if len(boxes) > 0:
                                        xyxy = boxes.xyxy.cpu().numpy()
                                        conf = boxes.conf.cpu().numpy()
                                        cls = boxes.cls.cpu().numpy()
                                        names = result.names  # Get class names from model
                                        
                                        for i in range(len(boxes)):
                                            x1, y1, x2, y2 = map(int, xyxy[i])
                                            confidence = float(conf[i])
                                            class_id = int(cls[i])
                                            
                                            # Get the actual class name from the model
                                            class_name = names.get(class_id, f'Class_{class_id}')
                                            
                                            # Check if this detection indicates mask wearing
                                            # Adjust these conditions based on your model's class names
                                            if 'mask' in class_name.lower() and 'without' not in class_name.lower():
                                                mask_detected_in_frame = True
                                                color = (0, 255, 0)  # Green for mask
                                            else:
                                                color = (0, 0, 255)  # Red for no mask
                                            
                                            # Draw bounding box
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                            
                                            # Draw recognized name above box
                                            cv2.putText(annotated_frame, name_to_show, (x1, y1-30), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                                            
                                            # Draw model's own class label with confidence
                                            label = f'{class_name}: {confidence:.3f}'
                                            cv2.putText(annotated_frame, label, (x1, y1-10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        elif model_type == 'keras':
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_frame)
                            for (top, right, bottom, left) in face_locations:
                                face_crop = frame[top:bottom, left:right]
                                if face_crop.size > 0:
                                    try:
                                        face_resized = cv2.resize(face_crop, (64, 64))
                                        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                                        face_normalized = face_rgb.astype('float32') / 255.0
                                        face_batch = np.expand_dims(face_normalized, axis=0)
                                        predictions = model.predict(face_batch, verbose=0)
                                        class_id = np.argmax(predictions[0])
                                        confidence = float(predictions[0][class_id])
                                        
                                        if confidence < 0.3:
                                            continue
                                            
                                    except Exception as e:
                                        print(f"Keras model error: {e}")
                                        continue
                                    
                                    # Get the actual class name from the model
                                    # Define class names for your specific model
                                    keras_class_names = {
                                        0: "with_mask",
                                        1: "without_mask", 
                                        2: "mask_incorrect"
                                    }
                                    class_name = keras_class_names.get(class_id, f'Class_{class_id}')
                                    
                                    # Check if this detection indicates mask wearing (similar to YOLO's logic)
                                    if 'mask' in class_name.lower() and 'without' not in class_name.lower():
                                        mask_detected_in_frame = True
                                        color = (0, 255, 0)  # Green for mask
                                    else:
                                        color = (0, 0, 255)  # Red for no mask
                                    
                                    # Draw bounding box
                                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                                    
                                    # Draw recognized name above box
                                    cv2.putText(annotated_frame, name_to_show, (left, top-30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                                    
                                    # Draw model's own class label with confidence
                                    label = f'{class_name}: {confidence:.3f}'
                                    cv2.putText(annotated_frame, label, (left, top-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        # Step 3: 3-Second Timer for Mask
                        if mask_detected_in_frame:
                            if verification_timer[0] == 0:
                                last_verification_time[0] = current_time
                                verification_timer[0] = 1
                                status_label.configure(text='✓ Mask detected! Stay still for 3 seconds...', text_color='#22d3ee')
                            else:
                                elapsed = current_time - last_verification_time[0]
                                remaining = 3.0 - elapsed
                                if remaining > 0:
                                    status_label.configure(text=f'✓ Stay still... {remaining:.1f}s remaining', text_color='#22d3ee')
                                else:
                                    status_label.configure(text='🎉 ATTENDANCE MARKED! You may leave.', text_color='#00ff00')
                                    mask_confirmed[0] = True
                                    run_detection[0] = False
                                    cap.release()
                                    log_attendance(emp_name)
                                    self._show_attendance_info_card(emp_name)
                                    return
                            mask_confirmed[0] = True
                        else:
                            verification_timer[0] = 0
                            status_label.configure(text='⚠ No proper mask detected. Please wear your mask correctly.', text_color='#ef4444')
                            mask_confirmed[0] = False
                    else:
                        verification_timer[0] = 0
                        mask_confirmed[0] = False
                    from PIL import Image, ImageTk
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    preview_box.imgtk = imgtk
                    preview_box.configure(image=imgtk, text="")
                except Exception as e:
                    status_label.configure(text=f'Error: {str(e)[:30]}...', text_color='#ef4444')
                    from PIL import Image, ImageTk
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    preview_box.imgtk = imgtk
                    preview_box.configure(image=imgtk, text="")
            else:
                status_label.configure(text='Camera read error', text_color='#ef4444')
            if run_detection[0]:
                preview_box.after(30, detection_loop)
        detection_loop()
        return mask_confirmed[0]

    def _show_attendance_info_card(self, name):
        # Find employee record (support both formats)
        emp = None
        for rec in self.db:
            if len(rec) == 2 and rec[0] == name:
                emp = {'id': '', 'name': rec[0], 'gender': '', 'issue': '', 'expiry': '', 'last_att': '', 'photo': 'default_male.png', 'encoding': rec[1]}
                break
            elif len(rec) >= 8 and rec[1] == name:
                emp = {'id': rec[0], 'name': rec[1], 'gender': rec[2], 'issue': rec[3], 'expiry': rec[4], 'last_att': rec[5], 'photo': rec[6], 'encoding': rec[7]}
                break
        if emp is None:
            return
        # Create a frame docked to the far right, fixed position
        info_card = ctk.CTkFrame(self.root, fg_color='#181c3a', corner_radius=18, width=420, height=600)
        info_card.place(relx=1.0, rely=0.5, anchor='e', x=-60)
        company_label = ctk.CTkLabel(info_card, text='TARUMT', font=('Segoe UI', 28, 'bold'), text_color='#2563eb')
        company_label.pack(pady=(32, 8))
        from PIL import Image, ImageTk
        photo_path = emp['photo']
        img = None
        if photo_path and os.path.exists(photo_path):
            try:
                img = Image.open(photo_path).resize((180, 180))
            except Exception:
                img = None
        if img is None:
            img_dir = os.path.join('datasets', 'employee_images')
            if emp['gender'] and emp['gender'].lower() == 'female':
                default_path = os.path.join(img_dir, 'default_female.jpg')
            else:
                default_path = os.path.join(img_dir, 'default_male.jpeg')
            try:
                img = Image.open(default_path).resize((180, 180))
            except Exception:
                img = Image.new('RGB', (180, 180), color='#222')
        photo_imgtk = ImageTk.PhotoImage(img)
        photo_label = ctk.CTkLabel(info_card, image=photo_imgtk, text='')
        photo_label.image = photo_imgtk
        photo_label.pack(pady=(0, 12))
        name_label = ctk.CTkLabel(info_card, text=emp['name'], font=('Segoe UI', 26, 'bold'), text_color='#fff')
        name_label.pack()
        id_label = ctk.CTkLabel(info_card, text=f"ID: {emp['id']}", font=('Segoe UI', 18), text_color='#b3b8d1')
        id_label.pack()
        info_text = f"Issue: {emp['issue']}   Expiry: {emp['expiry']}"
        ctk.CTkLabel(info_card, text=info_text, font=('Segoe UI', 16), text_color='#b3b8d1').pack(pady=(8, 0))
        if emp.get('last_att'):
            ctk.CTkLabel(info_card, text=f"Last attendance: {emp['last_att']}", font=('Segoe UI', 15), text_color='#b3b8d1').pack()
        if emp.get('role'):
            ctk.CTkLabel(info_card, text=f"Role: {emp['role']}", font=('Segoe UI', 15), text_color='#b3b8d1').pack()
        # Attendance taken message
        attendance_label = ctk.CTkLabel(info_card, text=f"Attendance taken for {emp['name']}!", font=('Segoe UI', 22, 'bold'), text_color='#22d3ee')
        attendance_label.pack(pady=(18, 0))
        # Button to return to main menu
        ctk.CTkButton(info_card, text='Back to Main Menu', font=('Segoe UI', 15, 'bold'), fg_color='#2563eb', hover_color='#1d4ed8', width=180, corner_radius=10, command=self.main_menu).pack(pady=(24, 0))

    def view_attendance(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.configure(bg='#f5f6fa')

        frame = ctk.CTkFrame(self.root, fg_color='#ffffff', corner_radius=18)
        frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.7, relheight=0.7)

        title = ctk.CTkLabel(frame, text='Attendance Records', font=('Segoe UI', 22, 'bold'), text_color='#2563eb')
        title.pack(pady=(32, 10))

        text_widget = ctk.CTkTextbox(frame, font=('Segoe UI', 12), width=600, height=400)
        text_widget.pack(padx=24, pady=10, fill='both', expand=True)
        if not os.path.exists(ATTENDANCE_LOG):
            text_widget.insert('1.0', 'No attendance records found.')
        else:
            try:
                with open(ATTENDANCE_LOG, 'r', newline='') as f:
                    reader = csv.reader(f)
                    records = list(reader)
                if len(records) <= 1:
                    text_widget.insert('1.0', 'No attendance records found.')
                else:
                    if records[0] == ['Name', 'Timestamp']:
                        text_widget.insert('1.0', f"{'Name':<20} {'Timestamp'}\n")
                        text_widget.insert('end', '-' * 50 + '\n')
                        records = records[1:]
                    for record in reversed(records[-50:]):
                        if len(record) >= 2:
                            text_widget.insert('end', f"{record[0]:<20} {record[1]}\n")
            except Exception as e:
                text_widget.insert('1.0', f'Error reading attendance log: {str(e)}')
        text_widget.configure(state='disabled')
        ctk.CTkButton(frame, text='Back', fg_color='#6c757d', text_color='white', command=self.admin_menu, width=150, height=40).pack(pady=20)

if __name__ == '__main__':
    root = ctk.CTk()
    root.title("Attendance System")

    # ✅ Windowed fullscreen (maximized with title bar & controls)
    root.state('zoomed')

    app = AttendanceApp(root)
    root.mainloop()
