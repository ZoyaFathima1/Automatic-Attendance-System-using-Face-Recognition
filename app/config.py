import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or '09251ef179b134ae99f35b707310fc2b'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # More config, e.g., database URI, upload folders, etc.
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///yourdatabase.db'
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
