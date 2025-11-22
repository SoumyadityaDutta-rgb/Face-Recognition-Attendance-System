# 🎯 Face Recognition Attendance System

A real-time face recognition-based attendance system built with Python, OpenCV, and the face_recognition library (powered by dlib's state-of-the-art deep learning models).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Real-time Face Recognition**: Uses dlib's ResNet-based 128D face embeddings for accurate recognition
- **Automatic Attendance Logging**: Marks attendance automatically when a known face is detected
- **Cooldown System**: Prevents duplicate entries with a 5-second recognition cooldown
- **CSV Export**: Saves attendance records with name, time, and date in CSV format
- **Efficient Processing**: Resizes frames for faster processing without compromising accuracy
- **Easy Training**: Simply add images to the `images/` folder and run the script
- **Persistent Encodings**: Face encodings are saved and reused, eliminating the need to retrain every time

## 🔍 How It Works

1. **Training Phase**:
   - Reads images from the `images/` folder
   - Detects faces using HOG (Histogram of Oriented Gradients)
   - Generates 128-dimensional face encodings using dlib's ResNet model
   - Saves encodings to `models/encodings.pkl` for future use

2. **Recognition Phase**:
   - Captures video from webcam in real-time
   - Detects faces in each frame
   - Compares detected faces with known encodings
   - Marks attendance when a match is found (with cooldown to prevent duplicates)
   - Displays live video feed with bounding boxes and names

## 🛠️ Prerequisites

- **Python**: 3.8 or higher
- **Webcam**: Required for real-time face detection
- **Operating System**: Windows, macOS, or Linux
- **CMake**: Required for building dlib (install via `pip install cmake` or system package manager)

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/SoumyadityaDutta-rgb/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv deepface_env
.\deepface_env\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv deepface_env
source deepface_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for Windows users**: If you encounter issues installing `dlib`, you may need to install it from a pre-built wheel:

```powershell
# Download the appropriate .whl file from https://github.com/z-mahmud22/Dlib_Windows_Python3.x
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
```

### Step 4: Add Training Images

1. Create an `images/` folder in the project directory (if not exists)
2. Add images of people you want to recognize
3. **Naming Convention**: Name files as `PersonName.jpg` or `PersonName_1.jpg`
   - Example: `JohnDoe.jpg`, `JaneSmith_1.jpg`, `JaneSmith_2.jpg`
   - The system uses the part before the underscore as the person's name

**Image Guidelines**:
- Use clear, front-facing photos
- Good lighting conditions
- One face per image
- Supported formats: `.jpg`, `.jpeg`, `.png`

## 🚀 Usage

### Running the Attendance System

```bash
python attendaceproject.py
```

### First Run (Training)

On the first run, the system will:
1. Scan the `images/` folder
2. Detect faces in each image
3. Generate and save face encodings
4. Start the webcam for real-time recognition

### Subsequent Runs

The system will:
1. Load pre-saved encodings from `models/encodings.pkl`
2. Start the webcam immediately
3. Begin recognizing faces and marking attendance

### Controls

- **Press 'q'**: Quit the application
- **Green Box**: Known person detected
- **Red Box**: Unknown person detected

### Re-training the Model

To add new people or update encodings:

1. Add new images to the `images/` folder
2. Delete `models/encodings.pkl` (or set `force_reencode = True` in the code)
3. Run the script again

## 📁 Project Structure

```
attendance-system/
│
├── attendaceproject.py       # Main application file
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
│
├── images/                    # Training images folder
│   ├── JohnDoe.jpg
│   ├── JaneSmith_1.jpg
│   └── ...
│
├── models/                    # Generated models folder
│   └── encodings.pkl          # Saved face encodings (auto-generated)
│
└── Attendance.csv             # Attendance log (auto-generated)
```

## ⚙️ Configuration

You can modify these parameters in `attendaceproject.py`:

```python
# Tolerance for face matching (lower = stricter)
TOLERANCE = 0.45  # Range: 0.0 to 1.0 (default: 0.45)

# Frame resize scale for faster processing
FRAME_RESIZE_SCALE = 0.25  # Range: 0.1 to 1.0 (default: 0.25)

# Cooldown between recognitions (in seconds)
RECOGNITION_COOLDOWN = 5  # Default: 5 seconds
```

### Parameter Tuning:

- **TOLERANCE**: 
  - Lower (0.3-0.4): More strict, fewer false positives
  - Higher (0.5-0.6): More lenient, may increase false positives
  
- **FRAME_RESIZE_SCALE**:
  - Lower (0.2): Faster processing, slightly lower accuracy
  - Higher (0.5): Slower processing, better accuracy

- **RECOGNITION_COOLDOWN**:
  - Prevents the same person from being marked multiple times in quick succession

## 🐛 Troubleshooting

### Issue: "Cannot access webcam"
**Solution**: 
- Ensure your webcam is connected and not being used by another application
- Check webcam permissions in your OS settings

### Issue: "No face found in image"
**Solution**:
- Use clear, front-facing photos with good lighting
- Ensure the face is clearly visible and not obscured
- Try different images of the same person

### Issue: "dlib installation fails"
**Solution**:
- Install CMake: `pip install cmake`
- For Windows: Download pre-built wheel from [here](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)
- For macOS: `brew install cmake`

### Issue: Low recognition accuracy
**Solution**:
- Add multiple images of the same person from different angles
- Improve lighting conditions
- Adjust the `TOLERANCE` parameter
- Use higher quality images

### Issue: Slow performance
**Solution**:
- Decrease `FRAME_RESIZE_SCALE` (e.g., 0.2)
- Use a more powerful computer
- Close other resource-intensive applications

## 📊 Attendance Log Format

The system generates `Attendance.csv` with the following format:

```csv
Name,Time,Date
JOHNDOE,09:30:15,2025-11-22
JANESMITH,09:31:42,2025-11-22
```

## 🔒 Privacy & Security

- **Personal Data**: The `images/` folder and `Attendance.csv` are excluded from Git by default
- **Local Processing**: All face recognition happens locally on your machine
- **No Cloud Storage**: No data is sent to external servers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [dlib](http://dlib.net/) by Davis King
- [OpenCV](https://opencv.org/) for computer vision capabilities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ by Soumyaditya**

⭐ If you found this project helpful, please give it a star!
