# Face Mask Detection Attendance System 😷

A comprehensive biometric attendance system that combines face recognition with AI-powered mask detection for secure and automated attendance tracking.

## 🚀 Features

- **Face Recognition**: Secure employee identification using facial encodings
- **AI Mask Detection**: Support for YOLOv8, YOLOv11, and CNN models
- **Security Features**: Anti-spoofing, unauthorized person detection, multi-person alerts
- **Model Management**: Command-line interface for switching between AI models
- **Camera Flexibility**: Support for PC camera and IP camera (phone)
- **Admin Dashboard**: Employee management, attendance logs, database administration
- **Real-time Processing**: Live camera feed with FPS monitoring

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11 (tested)
- **Hardware**: Webcam or IP camera
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB free space

### Python Dependencies
```bash
pip install -r requirements.txt
```

## 🛠️ Installation

### 1. Download and Extract
- Download the project files to your desired location
- Extract to a folder (e.g., `C:\FaceMaskAttendance\`)

### 2. Install Dependencies
Open Command Prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

**If face_recognition installation fails:**
```bash
# Windows
pip install cmake
pip install dlib
pip install face-recognition

# Linux/Mac
sudo apt-get install cmake
pip install dlib
pip install face-recognition
```

### 3. Create Directory Structure
```bash
mkdir models
mkdir datasets
mkdir datasets\employee_images
```

### 4. Add AI Models
Place your trained models in the `models\` folder:
- **YOLO models**: `.pt` files (YOLOv8/YOLOv11)
- **Keras models**: `.keras` files (TensorFlow/CNN)

## 🎯 Usage Guide

### Starting the Application
```bash
python m.py
```

## 👨‍💼 Admin Functions

### Login
- **Password**: `123`
- Access: Admin Dashboard with full system control

### Employee Management

#### Adding New Employees
1. **Via Camera (Recommended)**:
   - Enter employee name and gender
   - Click "Add via Camera"
   - Capture 5 photos using spacebar
   - System automatically saves and creates face encoding

2. **Via File Upload**:
   - Select 5-10 clear face images
   - System processes and creates face encoding
   - Supports JPG, PNG, BMP formats

#### Managing Database
- **View Images**: See all captured photos for each employee
- **Edit Details**: Modify name, gender, photo, and other details
- **Delete Records**: Remove employees from system (includes image cleanup)

### Attendance Management
- **View Records**: Real-time attendance log with timestamps
- **Clear Activity**: Reset attendance history
- **Export Data**: CSV format for external analysis

## 👤 Employee Functions

### Face Recognition Login
1. Click "Employee Login"
2. Position face in camera view
3. Wait for green bounding box (recognition success)
4. Proceed to mask verification

### Mask Detection Process
1. **Step 1**: Identity verification (completed after face recognition)
2. **Step 2**: Mask verification
   - Wear mask properly (covering nose and mouth)
   - Stay still for 3 seconds when mask detected
   - Attendance automatically marked on success

## 🤖 AI Model Management

### Command Prompt Access
- **Password**: `456`
- **Location**: Available during mask detection phase

### Available Commands

#### Model Management
```bash
# List all available models
model list

# Switch to specific model
model use mask_yolov5.pt
model use best.pt
model use high_accuracy_face_mask_model.keras

# Show current model status
status

# Open model selection GUI
model list box
```

#### Camera Control
```bash
# Switch to PC camera
camera switch default

# Switch to IP camera (phone)
camera switch ip

# Custom IP camera
camera switch http://192.168.1.100:8080/video
```

#### General Commands
```bash
# Show available commands
help

# Clear terminal
clear

# Echo text
echo Hello World
```

### Supported AI Models

#### YOLO Models (.pt files)
- **YOLOv8**: Fast, accurate object detection
- **YOLOv11**: Enhanced accuracy with better small object detection
- **Classes**: `mask`, `no-mask`, `mask_incorrect`
- **Performance**: ~30 FPS

#### Keras Models (.keras files)
- **CNN Architecture**: Detailed face analysis
- **Input Size**: 64×64 pixels
- **Classes**: `with_mask`, `without_mask`, `mask_incorrect`
- **Performance**: ~20 FPS, higher precision

## 📱 Camera Setup

### PC Camera
- **Default**: Built-in webcam (camera source = 0)
- **Auto-detection**: System automatically uses available camera

### IP Camera (Phone)
1. **Install IP Camera App** on your phone (e.g., IP Webcam)
2. **Connect to same WiFi** as your computer
3. **Start camera server** on phone
4. **Use command**: `camera switch ip` (uses default IP)
5. **Custom IP**: `camera switch http://YOUR_PHONE_IP:PORT/video`

## 🔒 Security Features

### Anti-Spoofing Protection
- **Live Detection**: Only processes live camera feed
- **Identity Verification**: Continuous person verification during process
- **Multi-Person Detection**: Blocks attempts with multiple people

### Unauthorized Access Prevention
- **Person Substitution Alert**: Detects if someone else replaces authorized user
- **Persistent Monitoring**: Maintains security throughout entire process
- **Timer-Based Confirmation**: Requires sustained presence for attendance

### Data Security
- **Local Storage**: All data stored locally (no cloud dependency)
- **Encrypted Encodings**: Face data stored as mathematical vectors
- **Access Control**: Password-protected admin functions

## 📊 Performance Monitoring

### Real-Time FPS Display
During mask detection, terminal shows:
```
FPS: 28.5 | Processing: 25.3ms
```

### System Status
- **Model Loading**: Real-time status updates
- **Detection Results**: Confidence scores and classifications
- **Error Handling**: Clear error messages and recovery options

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Working
```bash
# Try different camera source
camera switch default

# Check camera permissions in Windows Settings
# Restart application
```

#### Model Loading Errors
```bash
# Check model file exists in models/ folder
# Verify model format (.pt for YOLO, .keras for CNN)
# Try different model: model use [different_model.pt]
```

#### Face Recognition Issues
- **Lighting**: Ensure good lighting on face
- **Position**: Face camera directly, avoid extreme angles
- **Distance**: Stay 1-2 feet from camera
- **Multiple People**: Only one person in frame

#### Installation Problems
```bash
# Update pip
python -m pip install --upgrade pip

# Install dependencies individually
pip install ultralytics
pip install opencv-python
pip install customtkinter
pip install face-recognition
```

### Performance Optimization

#### For Better Speed
- Use YOLO models (.pt files)
- Ensure good lighting (reduces processing time)
- Close other camera applications
- Use PC camera instead of IP camera for lower latency

#### For Better Accuracy
- Use CNN models (.keras files) for detailed analysis
- Ensure proper lighting conditions
- Train custom models with your specific environment
- Use multiple angles during employee registration

## 📁 File Structure

```
project_folder/
│
├── m.py                           # Main application file
├── requirements.txt               # Python dependencies
├── employee_db.csv               # Employee database
├── attendance.csv                # Attendance records
│
├── models/                       # AI model files
│   ├── mask_yolov5.pt           # Default YOLO model
│   ├── best.pt                   # Custom YOLOv11 model
│   └── high_accuracy_face_mask_model.keras  # CNN model
│
├── datasets/
│   └── employee_images/          # Employee photos
│       ├── John_Doe/            # Individual employee folders
│       │   ├── cam_1.jpg        # Camera captures
│       │   └── upload_1.jpg     # Uploaded images
│       └── default_male.png     # Default profile images
│
└── notebooks/                    # Training notebooks (optional)
    └── Yolov8_Face_Mask.ipynb   # Model training reference
```

## 🎯 Default Settings

### Passwords
- **Admin Login**: `123`
- **Command Prompt**: `456`

### Model Configuration
- **Default Model**: `mask_yolov5.pt`
- **Confidence Threshold**: 0.3
- **IoU Threshold**: 0.45
- **Face Match Threshold**: 0.45

### Camera Settings
- **Default Source**: PC Camera (0)
- **IP Camera**: `http://192.168.100.28:8080/video`
- **Resolution**: 640×480 (auto-resized)
- **FPS Target**: 30

### Timer Settings
- **Mask Verification**: 3 seconds
- **Identity Verification**: 2 seconds  
- **No Person Timeout**: 20 seconds

## 📞 Support

### System Requirements Check
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

### Model Testing
```bash
# Test YOLO model
python -c "from ultralytics import YOLO; m=YOLO('models/mask_yolov5.pt'); print('YOLO OK')"

# Test face recognition
python -c "import face_recognition; print('Face recognition OK')"
```

## 🎉 Quick Start Guide

1. **Install**: Run `pip install -r requirements.txt`
2. **Run**: Execute `python m.py`
3. **Admin Setup**: Login with password `123`, add employees
4. **Employee Use**: Click "Employee Login", complete face recognition and mask detection
5. **Model Switch**: Use command prompt (password `456`) to change AI models

## 📈 Advanced Features

### Custom Model Training
- Refer to `notebooks/Yolov8_Face_Mask.ipynb` for training procedures
- Support for custom datasets and classes
- Model export in compatible formats

### Integration Options
- CSV export for external systems
- REST API potential for enterprise integration
- Database migration tools for larger deployments

---

**Note**: This system is designed for educational and small-scale deployment purposes. For enterprise use, consider additional security measures and compliance requirements.