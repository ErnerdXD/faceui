import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time

# Load YOLOv11 model
model = YOLO(r"C:\\Users\\Ernest\\Desktop\\ai2\\actual\\models\\dada.pt")

# --- Camera setup ---
cap = cv2.VideoCapture(0)  # Local PC webcam
# cap = cv2.VideoCapture("http://192.168.100.28:8080/video")  # Use phone/IP camera

root = tk.Tk()
root.title("YOLOv11 Mask Detector")

lbl = tk.Label(root)
lbl.pack()

status_lbl = tk.Label(root, text="", font=("Arial", 16), fg="blue")
status_lbl.pack(pady=10)

running = True
frame_lock = threading.Lock()
processed_frame = None

# --- Mask detection timer ---
mask_start_time = None
mask_confirmed = False


def video_capture():
    """Capture frames and run YOLO detection in background."""
    global processed_frame, mask_start_time, mask_confirmed, running

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, stream=True, imgsz=320)
        mask_detected = False

        for r in results:
            frame = r.plot()
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label.lower() == "mask":  # Check if mask detected
                    mask_detected = True

        # Handle mask timer
        if mask_detected and not mask_confirmed:
            if mask_start_time is None:
                mask_start_time = time.time()  # Start countdown
            else:
                elapsed = time.time() - mask_start_time
                if elapsed >= 2:  # Mask worn for 2s
                    mask_confirmed = True
                    status_lbl.config(text="Congrats! You are wearing a mask 🎉", fg="green")
                    popup_congrats()
                    root.after(2000, on_close)  # Auto close after 2s
        else:
            mask_start_time = None  # Reset if mask not detected

        # Store frame for GUI
        with frame_lock:
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def update_gui():
    """Update Tkinter label smoothly with latest frame + status."""
    global processed_frame, mask_start_time, mask_confirmed

    if processed_frame is not None:
        with frame_lock:
            img = Image.fromarray(processed_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl.imgtk = imgtk
            lbl.configure(image=imgtk)

    # Show countdown if running
    if mask_start_time and not mask_confirmed:
        elapsed = time.time() - mask_start_time
        remaining = round(2 - elapsed, 1)
        if remaining > 0:
            status_lbl.config(text=f"Keep wearing mask... {remaining:.1f}s left", fg="blue")

    if running:
        root.after(30, update_gui)


def popup_congrats():
    """Show popup congratulation message."""
    popup = tk.Toplevel(root)
    popup.title("Success")
    tk.Label(popup, text="🎉 Congrats! You are wearing a mask 🎉",
             font=("Arial", 16), fg="green").pack(padx=20, pady=20)
    tk.Button(popup, text="OK", command=popup.destroy).pack(pady=10)


def on_close():
    """Clean shutdown."""
    global running
    running = False
    cap.release()
    root.destroy()


quit_btn = tk.Button(root, text="Quit", command=on_close, bg="red", fg="white")
quit_btn.pack(pady=5)

# Start capture thread
threading.Thread(target=video_capture, daemon=True).start()
update_gui()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
