import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///attendance.db'
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'instance', 'student_images')
