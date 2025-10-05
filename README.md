🧠 Face Recognition Attendance System
A Flask-powered web application for automated, secure attendance using cutting-edge face recognition.
Built for real classrooms and robust against data leakage, it features admin, faculty, and student dashboards.

🚀 Tech Stack
🐍 Python 3.x

⚡ Flask (Web Framework)

🤖 TensorFlow / Keras (FaceNet Model)

📷 OpenCV

📊 NumPy / Pandas

🗄️ SQLite / MySQL (for attendance storage)

🎯 Features
🔎 Real-time face detection & recognition (MTCNN + FaceNet)

📝 Automatic, accurate attendance marking

🛠️ Data augmentation, strict leakage prevention

👑 Admin dashboard: manage users & subjects, configure system

👩‍🏫 Faculty dashboard: start/stop sessions, export to Excel, monitor stats

👨‍🎓 Student dashboard: attendance view, detailed graphs for each subject

📉 Visual analytics: confusion matrix, per-class metrics, accuracy curves

🛠️ Installation & Setup
1️⃣ Clone the Repository
bash
git clone https://github.com/<your-username>/face-recognition-attendance.git
cd face-recognition-attendance
2️⃣ Install Requirements
bash
pip install -r requirements.txt
3️⃣ Configure Your Environment
Edit config.py for DB, server, and upload directories as needed.

4️⃣ Initialize Database
bash
flask db init
flask db migrate
flask db upgrade
5️⃣ Run the Application
bash
flask run

# Visit http://localhost:5000

👥 User Roles
🔒 Admin:
Create students/faculty, assign subjects, oversee attendance

🧑‍🏫 Faculty:
Start/stop sessions, download attendance, monitor analytics

🧑‍🎓 Student:
View own attendance, subject-wise breakdown, charts

📚 Directory Structure
text
attendance-system/
├── app/
│ ├── models.py
│ ├── utils.py
│ ├── admin/
│ ├── faculty/
│ ├── student/
│ └── templates/
├── static/
│ ├── test_images/
│ └── test_labels.csv
├── instance/
│ ├── student_images/
│ └── train/, val/, augmented/
├── instance_augmented/
├── requirements.txt
├── config.py
├── README.md
📊 Evaluation & Analytics
⚡ Pre-split and augmentation for leakage-free experiments

📁 Independent manual test set for unbiased real-world performance

🏆 Fast evaluation pipeline with cached embeddings

🧾 Downloadable Excel attendance, interactive graphs per user
