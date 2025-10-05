from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    # Load configuration from 'config.py' Config class
    app.config.from_object('config.Config')

    # Ensure UPLOAD_FOLDER exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Set secret key for session management (replace with a secure secret in production)
    if not app.config.get('SECRET_KEY'):
        app.secret_key = os.urandom(24)
    else:
        app.secret_key = app.config['SECRET_KEY']

    # Initialize database
    db.init_app(app)

    with app.app_context():
        from . import routes

        # Register blueprints including 'main_bp' for landing page
        app.register_blueprint(routes.main_bp)
        app.register_blueprint(routes.admin_bp)
        app.register_blueprint(routes.attendance_bp)
        app.register_blueprint(routes.faculty_bp)
        app.register_blueprint(routes.student_bp)
        app.register_blueprint(routes.testdata_bp)

        # Create database tables if not existing
        db.create_all()

    return app
