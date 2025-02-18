# Attendance System using Face Recognition

## Overview
This project is a face recognition–based attendance system. It uses your webcam to detect and recognize faces in real time, logs the attendance in a uniquely named CSV file, and saves images of unrecognized (mismatched) faces in a separate folder. The log file includes the following details:
- **Date & Time**: Based on local time in the format `YYYY-MM-DD____HH-MM-SS_AM/PM`.
- **Name**: The recognized face's name (if a match is found), "Unknown" if the face does not match any in the folder, or "NF" if no face is detected.
- **Detected**: A "Yes" if a face is detected (even if it is a mismatch), and "No" if no face is detected.
- **Description**: A brief description of the event (e.g., "Face Matched" or details about the mismatch).

The program is designed to record exactly one log entry per run:
- If a face is detected within the first 5 seconds, it logs that face (or mismatch) and exits 5 seconds after detection.
- If no face is detected within 10 seconds, it logs "No Face Detected" and exits.

## Features
- **Face Recognition:** Compares captured faces with known images using the `face_recognition` library.
- **Automated Logging:** Generates a new log file for each run using the current date and time.
- **Mismatch Handling:** Saves images of mismatched faces in a dedicated folder (`mismatches`) and logs the event.
- **Timed Exit:** Exits 5 seconds after a face is detected or 10 seconds if no face is detected.
- **Cross-Platform Setup:** Developed on Windows; uses a virtual environment for dependency management.

## Installation

### Prerequisites
- Python 3.10 or 3.11 (recommended for compatibility with `face_recognition` and `dlib`).
- A webcam.

### Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/attendance-system.git
   cd attendance-system
2. Create a Virtual Environment:
   ```bash
    python -m venv venv

3.Activate the Virtual Environment:

   On Windows:
  ```bash
  venv\Scripts\activate
```
  On macOS/Linux:
   ```bash
    source venv/bin/activate
```
Install Dependencies:

    pip install -r requirements.txt

4. Usage

Run the attendance system with:
```bash
python attendance_system.py
```
