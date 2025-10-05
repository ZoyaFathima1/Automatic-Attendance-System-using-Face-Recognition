from flask import Blueprint, render_template, request, redirect, url_for, jsonify, current_app, session, flash, send_file, Response  # existing
from .models import Student, Faculty, Subject, FacultySubject, AttendanceSession, AttendanceRecord, Admin  # existing
from . import db  # existing
from .utils import recognize_faces_in_frame, load_embedding_from_db  # existing

# New/ensure present
import os
import base64
import glob
import shutil
import json
import cv2
import time
import io
import pandas as pd
import numpy as np
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import datetime as py_datetime  # optional alias to disambiguate

############################## MAIN ROUTES #############################################
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def landing_page():
    return render_template('landing.html')  # Render your Bootstrap template stored as landing.html

@main_bp.route('/analytics')
def show_analytics():
    with current_app.app_context():
        csv_path = os.path.join(current_app.root_path, 'static', 'test_labels.csv')
        test_labels = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    test_labels.append((row['image_path'], row['usn']))
        except FileNotFoundError:
            flash("Test labels CSV file not found.")
            return redirect(url_for('main.landing_page'))

        # Load known embeddings
        students = Student.query.all()
        known_embeddings_dict = {}
        for student in students:
            emb = load_embedding_from_db(student)
            if emb is not None:
                known_embeddings_dict[student.usn] = emb

        y_true = []
        y_pred = []

        for image_path, true_usn in test_labels:
            full_img_path = os.path.join(current_app.root_path, 'static', image_path)
            if not os.path.exists(full_img_path):
                # Skip missing images silently or log if preferred
                continue

            img = np.array(Image.open(full_img_path).convert('RGB'))
            results = recognize_faces_in_frame(img, known_embeddings_dict)

            if results and results[0][0] is not None:
                pred_usn = results[0][0]
            else:
                pred_usn = 'Unknown'  # Use string label to avoid None

            y_true.append(true_usn if true_usn is not None else 'Unknown')
            y_pred.append(pred_usn)

        labels = list(known_embeddings_dict.keys()) + ['Unknown']

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Plot confusion matrix heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted USN")
        plt.ylabel("Actual USN")
        plt.title("Confusion Matrix: Face Recognition")
        plt.tight_layout()

        cm_path = os.path.join(current_app.root_path, 'static', 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

        # Calculate metrics with macro average and zero division handling
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)

        return render_template(
            'analytics.html',
            accuracy=f"{accuracy:.4f}",
            precision=f"{precision:.4f}",
            recall=f"{recall:.4f}",
            f1_score=f"{f1:.4f}"
        )



############################# ADMIN ROUTES ####################################
# app/admin/routes.py

from flask import Blueprint, render_template, request, redirect, url_for, jsonify, current_app, session, flash
from app.models import Student, Faculty, Subject, FacultySubject, AttendanceSession, AttendanceRecord, Admin
from app import db
from werkzeug.security import check_password_hash, generate_password_hash
import os
import base64
import glob
import shutil
import csv

# Single blueprint declaration
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

def norm_usn(u: str) -> str:
    return (u or '').strip().upper()

# Dashboard
@admin_bp.route('/')
def admin_dashboard():
    students = Student.query.all()
    subjects = Subject.query.all()         # For delete subject/select subject
    faculties = Faculty.query.all()        # For delete faculty/select faculty
    return render_template('admin.html', students=students, subjects=subjects, faculties=faculties)


# Login
@admin_bp.route('/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = (request.form.get('password') or '').strip()
        admin = Admin.query.filter_by(username=username).first()
        if admin and check_password_hash(admin.password, password):
            session['admin_id'] = admin.id
            flash("Logged in successfully.", 'success')
            return redirect(url_for('admin.admin_dashboard'))
        else:
            flash("Invalid username or password.", 'error')
    return render_template('admin_login.html')

# Add student
@admin_bp.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        usn = norm_usn(request.form.get('usn'))
        if not name or not usn:
            flash('Name and USN are required.', 'error')
            return redirect(url_for('admin.add_student'))

        exists = Student.query.filter_by(usn=usn).first()
        if exists:
            flash(f'Student with USN {usn} already exists. You can capture more images.', 'error')
            return redirect(url_for('admin.capture_images', usn=usn))

        new_student = Student(name=name, usn=usn)
        db.session.add(new_student)
        db.session.commit()

        folder_path = os.path.join(current_app.config['UPLOAD_FOLDER'], usn)
        os.makedirs(folder_path, exist_ok=True)
        flash(f'Student {name} ({usn}) added. Proceed to capture images.', 'success')
        return redirect(url_for('admin.capture_images', usn=usn))
    return render_template('add_student.html')

# Capture page
@admin_bp.route('/capture_images/<usn>')
def capture_images(usn):
    return render_template('capture_images.html', usn=norm_usn(usn))

# Save a captured image (no fixed limit; auto-increment filenames)
@admin_bp.route('/save_image/<usn>', methods=['POST'])
def save_image(usn):
    usn = norm_usn(usn)
    data = request.get_json(silent=True) or {}
    img_data = data.get('image')
    if not img_data:
        return jsonify({"status": "error", "message": "Missing image"}), 400

    # Decode base64
    try:
        b64 = img_data.split(',', 1)[1] if ',' in img_data else img_data
        img_bytes = base64.b64decode(b64, validate=True)
    except Exception:
        return jsonify({"status": "error", "message": "Invalid base64 image data"}), 400

    # Ensure folder
    folder = os.path.join(current_app.config['UPLOAD_FOLDER'], usn)
    os.makedirs(folder, exist_ok=True)

    # Auto-increment filename
    existing = sorted(glob.glob(os.path.join(folder, '*.jpg')) +
                      glob.glob(os.path.join(folder, '*.jpeg')) +
                      glob.glob(os.path.join(folder, '*.png')))
    next_idx = len(existing) + 1
    img_path = os.path.join(folder, f"{next_idx}.jpg")

    try:
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to save image: {e}"}), 500

    return jsonify({"status": "success", "path": img_path, "count": next_idx})

#edit existing student
@admin_bp.route('/edit_student', methods=['GET', 'POST'])
def edit_student():
    if request.method == 'POST':
        usn = (request.form.get('usn') or '').strip().upper()
        student = Student.query.filter_by(usn=usn).first()
        if student:
            # Redirect to the capture page to add more images
            return redirect(url_for('admin.capture_images', usn=usn))
        else:
            flash(f'Student with USN {usn} not found.', 'error')
    # Render a simple form to enter the USN
    students = Student.query.all()  # You can pass this to display a student list if desired
    return render_template('edit_student.html', students=students)

# Finalize enrollment (recompute embeddings)
@admin_bp.route('/finalize_enrollment/<usn>', methods=['POST'])
def finalize_enrollment(usn):
    from app.utils import compute_and_store_embeddings
    usn = norm_usn(usn)
    folder = os.path.join(current_app.config['UPLOAD_FOLDER'], usn)
    if not os.path.isdir(folder):
        return jsonify({"status": "error", "message": "No folder for USN"}), 400

    ok = compute_and_store_embeddings(usn, folder, Student, db, auto_clear=True, min_prob=0.95)
    if not ok:
        return jsonify({"status": "error", "message": "No valid faces found"}), 400
    return jsonify({"status": "success", "message": "Embedding updated"})

# Add subject
@admin_bp.route('/add_subject', methods=['GET', 'POST'])
def add_subject():
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        if not name:
            flash('Subject name is required.', 'error')
            return redirect(url_for('admin.add_subject'))
        db.session.add(Subject(name=name))
        db.session.commit()
        flash('Subject added.', 'success')
        return redirect(url_for('admin.admin_dashboard'))
    return render_template('add_subject.html')

#delete subject (also cleans assignments and attendance)
@admin_bp.route('/delete_subject', methods=['POST'])
def delete_subject():
    subject_id = request.form.get('subject_id')
    if not subject_id:
        flash("Please provide a subject to delete.", "error")
        return redirect(url_for('admin.admin_dashboard'))

    subject = Subject.query.get(subject_id)
    if not subject:
        flash("No subject found.", "error")
        return redirect(url_for('admin.admin_dashboard'))

    try:
        # Remove all assignments to faculty
        FacultySubject.query.filter_by(subject_id=subject_id).delete()
        # Remove ALL attendance sessions/records for this subject
        sessions = AttendanceSession.query.filter_by(subject_id=subject_id).all()
        for s in sessions:
            AttendanceRecord.query.filter_by(session_id=s.id).delete()
            db.session.delete(s)
        db.session.delete(subject)
        db.session.commit()
        flash(f"Subject {subject.name} deleted with all related assignments and attendance.", "success")
    except Exception as e:
        db.session.rollback()
        flash("Error deleting subject: " + str(e), "error")
    return redirect(url_for('admin.admin_dashboard'))


# Assign subjects to faculty
@admin_bp.route('/assign_subjects/<int:faculty_id>', methods=['GET', 'POST'])
def assign_subjects(faculty_id):
    faculty = Faculty.query.get_or_404(faculty_id)
    subjects = Subject.query.all()
    if request.method == 'POST':
        selected_subject_ids = request.form.getlist('subjects')
        for sid in selected_subject_ids:
            if not FacultySubject.query.filter_by(faculty_id=faculty.id, subject_id=int(sid)).first():
                db.session.add(FacultySubject(faculty_id=faculty.id, subject_id=int(sid)))
        db.session.commit()
        flash('Subjects assigned.', 'success')
        return redirect(url_for('admin.admin_dashboard'))
    return render_template('assign_subjects.html', faculty=faculty, subjects=subjects)

# Add faculty
@admin_bp.route('/add_faculty', methods=['GET', 'POST'])
def add_faculty():
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        username = (request.form.get('username') or '').strip()
        password = (request.form.get('password') or '').strip()

        if not name or not username or not password:
            error = "All fields are required"
            return render_template('add_faculty.html', error=error)

        if Faculty.query.filter_by(username=username).first():
            error = "Username already exists"
            return render_template('add_faculty.html', error=error)

        hashed_password = generate_password_hash(password)
        new_faculty = Faculty(name=name, username=username, password=hashed_password)
        db.session.add(new_faculty)
        db.session.commit()

        flash('Faculty added. Assign subjects.', 'success')
        return redirect(url_for('admin.assign_subjects', faculty_id=new_faculty.id))
    return render_template('add_faculty.html')

#delete faculty (also cleans assignments and attendance)
@admin_bp.route('/delete_faculty', methods=['POST'])
def delete_faculty():
    faculty_id = request.form.get('faculty_id')
    if not faculty_id:
        flash("Please provide a faculty ID to delete.", "error")
        return redirect(url_for('admin.admin_dashboard'))

    faculty = Faculty.query.get(faculty_id)
    if not faculty:
        flash("No faculty found.", "error")
        return redirect(url_for('admin.admin_dashboard'))

    try:
        # Remove subject assignments
        FacultySubject.query.filter_by(faculty_id=faculty_id).delete()
        # Remove sessions/attendance records led by this faculty
        sessions = AttendanceSession.query.filter_by(faculty_id=faculty_id).all()
        for s in sessions:
            AttendanceRecord.query.filter_by(session_id=s.id).delete()
            db.session.delete(s)
        db.session.delete(faculty)
        db.session.commit()
        flash(f"Faculty {faculty.name} deleted with all related sessions and assignments.", "success")
    except Exception as e:
        db.session.rollback()
        flash("Error deleting faculty: " + str(e), "error")
    return redirect(url_for('admin.admin_dashboard'))

# Delete student (also cleans files and CSV)
@admin_bp.route('/delete_student', methods=['POST'])
def delete_student():
    usn = norm_usn(request.form.get('usn'))
    if not usn:
        flash('Please provide a USN to delete.', 'error')
        return redirect(url_for('admin.admin_dashboard'))

    student = Student.query.filter_by(usn=usn).first()
    if not student:
        flash(f'No student found with USN: {usn}', 'error')
        return redirect(url_for('admin.admin_dashboard'))

    try:
        # Delete attendance records
        attendance_records = AttendanceRecord.query.filter_by(student_id=student.id).all()
        for record in attendance_records:
            db.session.delete(record)

        # Delete student
        db.session.delete(student)
        db.session.commit()

        # Remove enrollment folder
        enroll_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], usn)
        if os.path.exists(enroll_folder):
            shutil.rmtree(enroll_folder)

        # Remove test images and prune CSV rows
        test_dir = os.path.join(current_app.root_path, 'static', 'test_images')
        csv_path = os.path.join(current_app.root_path, 'static', 'test_labels.csv')

        removed_files = 0
        if os.path.isdir(test_dir):
            for p in glob.glob(os.path.join(test_dir, f"{usn}_*.jpg")):
                try:
                    os.remove(p)
                    removed_files += 1
                except Exception:
                    pass

        if os.path.isfile(csv_path):
            rows_kept = []
            header = None
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) != 2:
                        continue
                    path, u = row[0].strip(), (row[1] or '').strip().upper()
                    if u != usn:
                        rows_kept.append([path, u])
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if header is not None and len(header) == 2:
                    writer.writerow(header)
                writer.writerows(rows_kept)

        flash(f'Deleted {usn}, related attendance records, enrollment folder, and {removed_files} test images; CSV pruned.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting student: {str(e)}', 'error')

    return redirect(url_for('admin.admin_dashboard'))

#Logout
@admin_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('admin.admin_login'))


############################# ATTENDANCE ROUTES ####################################

# Attendance Blueprint
attendance_bp = Blueprint('attendance', __name__, url_prefix='/attendance')

@attendance_bp.route('/')
def attendance():
    print("Full session:", dict(session))
    session_id = session.get('attendance_session_id')
    print("Attendance page session_id:", session_id)
    return render_template('attendance.html')


@attendance_bp.route('/session_status')
def session_status():
    session_id = request.args.get('session_id', type=int)
    if session_id is None:
        # No session specified, treat as ended
        return jsonify({"ended": True})
    session_obj = AttendanceSession.query.get(session_id)
    ended = session_obj.end_time is not None if session_obj else True
    return jsonify({"ended": ended})

@attendance_bp.route('/video_feed')
def video_feed():
    attendance_session_id = request.args.get('session_id', type=int)
    if attendance_session_id is None:
        # No active session, return empty response
        def empty_gen():
            yield b""
        return Response(empty_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    students = Student.query.filter(Student.embedding.isnot(None)).all()
    known_embeddings = {}
    from run import app as flask_app

    with flask_app.app_context():
        for student in students:
            emb = load_embedding_from_db(student)
            if emb is not None:
                known_embeddings[student.usn] = emb

    def generate(attendance_session_id, flask_app):
        with flask_app.app_context():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open webcam")
                yield b""
                return
            recognized_students = set()
            frame_count = 0
            try:
                while True:
                    session_obj = AttendanceSession.query.get(attendance_session_id)
                    if session_obj is None or session_obj.end_time is not None:
                        print("Attendance session ended, stopping video feed.")
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    recognized_faces = recognize_faces_in_frame(frame_rgb, known_embeddings)

                    if attendance_session_id:
                        for usn, _ in recognized_faces:
                            if usn is not None:
                                recognized_students.add(usn)
                        frame_count += 1
                        if frame_count % 50 == 0:
                            for usn in recognized_students:
                                student = Student.query.filter_by(usn=usn).first()
                                if student:
                                    attendance_record = AttendanceRecord.query.filter_by(
                                        session_id=attendance_session_id,
                                        student_id=student.id
                                    ).first()
                                    if attendance_record and not attendance_record.present:
                                        attendance_record.present = True
                                        db.session.add(attendance_record)
                                        print(f"Marked present: Student USN {student.usn}, Session {attendance_session_id}")
                            db.session.commit()
                            recognized_students.clear()

                    for usn, box in recognized_faces:
                        if usn is None or box is None:
                            continue
                        box = np.array(box)
                        if box.shape[0] != 4:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, usn, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    recognized_usns_only = [usn for usn, _ in recognized_faces if usn is not None]
                    y0, dy = 30, 30
                    for i, usn in enumerate(set(recognized_usns_only)):
                        y = y0 + i * dy
                        cv2.putText(frame, usn, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    ret, jpeg = cv2.imencode('.jpg', frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                    time.sleep(0.03)
            finally:
                cap.release()

    return Response(generate(attendance_session_id, flask_app), mimetype='multipart/x-mixed-replace; boundary=frame')


############################# FACULTY ROUTES ####################################

# Faculty Blueprint
faculty_bp = Blueprint('faculty', __name__, url_prefix='/faculty')

@faculty_bp.route('/login', methods=['GET', 'POST'])
def faculty_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        faculty = Faculty.query.filter_by(username=username).first()

        if faculty and check_password_hash(faculty.password, password):
            session['faculty_id'] = faculty.id
            flash('Logged in successfully.')
            return redirect(url_for('faculty.faculty_dashboard'))
        else:
            flash('Invalid username or password.')

    return render_template('faculty_login.html')

@faculty_bp.route('/')
def faculty_dashboard():
    faculty_id = session.get('faculty_id')
    if not faculty_id:
        return redirect(url_for('faculty.faculty_login'))

    subjects = db.session.query(Subject).join(FacultySubject).filter(
        FacultySubject.faculty_id == faculty_id).all()

    attendance_session_id = session.get('attendance_session_id')
    if attendance_session_id:
        att_session = AttendanceSession.query.get(attendance_session_id)
        if not att_session or att_session.end_time is not None:
            session.pop('attendance_session_id', None)

    subject_id = request.args.get('subject_id', type=int)
    if not subject_id:
        # No subject selected; render template with message or prompt
        return render_template(
            'faculty_dashboard.html',
            subjects=subjects,
            selected_subject=None,
            sessions=None,
            attendance_summary=None,
            message="Please select a subject to view attendance."
        )

    # Fetch sessions for subject & faculty
    sessions = AttendanceSession.query.filter_by(
        faculty_id=faculty_id, subject_id=subject_id
    ).order_by(AttendanceSession.start_time.desc()).all()

    # Get students who have embeddings for recognition
    students = Student.query.filter(Student.embedding.isnot(None)).all()
    
    # Preload attendance records for all sessions & students at once, indexed
    records = AttendanceRecord.query.filter(
        AttendanceRecord.session_id.in_([s.id for s in sessions]),
        AttendanceRecord.student_id.in_([stu.id for stu in students])
    ).all()
    # Create lookup: {(session_id, student_id): attendance_record}
    record_lookup = {(r.session_id, r.student_id): r for r in records}

    attendance_summary = {}
    for student in students:
        total_present = sum(
            1 for s in sessions
            if record_lookup.get((s.id, student.id)) and record_lookup[(s.id, student.id)].present
        )
        total_sessions = len(sessions)
        attendance_summary[student] = {
            'present': total_present,
            'absent': total_sessions - total_present
        }

    return render_template(
        'faculty_dashboard.html',
        subjects=subjects,
        selected_subject=subject_id,
        sessions=sessions,
        attendance_summary=attendance_summary,
        message=None
    )


@faculty_bp.route('/start_session', methods=['POST'])
def start_stop_session():
    action = request.form.get('action')
    subject_id = request.form.get('subject_id')
    faculty_id = session.get('faculty_id')

    if not faculty_id:
        return redirect(url_for('faculty.faculty_login'))

    try:
        subject_id = int(subject_id)
    except (ValueError, TypeError):
        flash('Invalid subject selected.')
        return redirect(url_for('faculty.faculty_dashboard'))

    if action == 'start':
        existing_session = AttendanceSession.query.filter_by(
            faculty_id=faculty_id, subject_id=subject_id, end_time=None).first()

        if existing_session:
            # Set session ID even if session already active
            session['attendance_session_id'] = existing_session.id
            flash('Session already active')
        else:
            new_session = AttendanceSession(
                faculty_id=faculty_id,
                subject_id=subject_id,
                start_time=datetime.datetime.utcnow(),
            )
            db.session.add(new_session)
            db.session.commit()

            students = Student.query.filter(Student.embedding.isnot(None)).all()
            for student in students:
                attendance_record = AttendanceRecord(
                    session_id=new_session.id,
                    student_id=student.id,
                    present=False
                )
                db.session.add(attendance_record)
            db.session.commit()

            session['attendance_session_id'] = new_session.id
            flash('Session started and attendance records initialized')

        # Redirect to attendance page after start
        return redirect(url_for('attendance.attendance'))

    elif action == 'stop':
        session_id = session.get('attendance_session_id')
        if session_id:
            att_session = AttendanceSession.query.get(session_id)
            if att_session and att_session.end_time is None:
                att_session.end_time = datetime.datetime.utcnow()
                db.session.commit()
                flash('Session stopped')
            session.pop('attendance_session_id', None)
            # Redirect back to dashboard after stop
            return redirect(url_for('faculty.faculty_dashboard'))
        else:
            flash('No active session to stop')
  
            return redirect(url_for('faculty.faculty_dashboard'))
        

@faculty_bp.route('/delete_attendance/<int:session_id>', methods=['POST'])
def delete_attendance(session_id):
    faculty_id = session.get('faculty_id')
    if not faculty_id:
        return redirect(url_for('faculty.faculty_login'))

    att_session = AttendanceSession.query.get(session_id)
    if not att_session or att_session.faculty_id != faculty_id:
        flash("Invalid session or permission denied")
        return redirect(url_for('faculty.faculty_dashboard'))

    # First delete attendance records
    AttendanceRecord.query.filter_by(session_id=session_id).delete()
    # Then delete the actual session
    db.session.delete(att_session)
    db.session.commit()
    flash('Attendance session and its records deleted.')
    return redirect(url_for('faculty.faculty_dashboard', subject_id=att_session.subject_id))


@faculty_bp.route('/export_attendance/<int:subject_id>')
def export_attendance(subject_id):
    faculty_id = session.get('faculty_id')
    if not faculty_id:
        return redirect(url_for('faculty.faculty_login'))

    attendance_sessions = AttendanceSession.query.filter_by(
        faculty_id=faculty_id,
        subject_id=subject_id
    ).order_by(AttendanceSession.start_time).all()

    student_ids = db.session.query(Student.id).filter(Student.embedding.isnot(None)).all()

    data = []
    session_dates = [att_session.start_time.strftime("%Y-%m-%d %H:%M") for att_session in attendance_sessions]
    columns = ['USN', 'Name'] + session_dates

    for sid, in student_ids:
        student = Student.query.get(sid)
        row = [student.usn, student.name]
        for att_session in attendance_sessions:
            attendance_record = AttendanceRecord.query.filter_by(
                session_id=att_session.id,
                student_id=student.id
            ).first()
            status = 'Present' if attendance_record and attendance_record.present else 'Absent'
            row.append(status)
        data.append(row)

    df = pd.DataFrame(data, columns=columns)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance')
    output.seek(0)

    filename = f"attendance_report_subject_{subject_id}.xlsx"
    return send_file(
        output,
        download_name=filename,
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


############################# STUDENT ROUTES ####################################

student_bp = Blueprint('student', __name__, url_prefix='/student')

@student_bp.route('/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        usn = request.form.get('usn')

        student = Student.query.filter_by(usn=usn).first()
        if student:
            session['student_id'] = student.id
            session['student_usn'] = student.usn
            flash('Logged in successfully.')
            return redirect(url_for('student.student_dashboard'))
        else:
            flash('Invalid USN. Please try again.')

    return render_template('student_login.html')


@student_bp.route('/dashboard')
def student_dashboard():
    student_id = session.get('student_id')
    if not student_id:
        flash("Please login first.")
        return redirect(url_for('student.student_login'))

    student = Student.query.get(student_id)
    if not student:
        flash("Invalid student session. Please login again.")
        return redirect(url_for('student.student_login'))

    # Get all subjects the student is associated with (or all subjects for simplicity)
    subjects = Subject.query.all()

    attendance_info = []
    for subject in subjects:
        sessions = AttendanceSession.query.filter_by(subject_id=subject.id).all()
        total_sessions = len(sessions)
        present_sessions = 0
        session_ids = [s.id for s in sessions]
        if session_ids:
            present_sessions = AttendanceRecord.query.filter(
                AttendanceRecord.student_id == student_id,
                AttendanceRecord.session_id.in_(session_ids),
                AttendanceRecord.present == True
            ).count()
        attendance_percent = (present_sessions / total_sessions * 100) if total_sessions > 0 else 0

        attendance_info.append({
            'subject': subject,
            'present': present_sessions,
            'total': total_sessions,
            'percent': round(attendance_percent, 2)
        })

    return render_template('student_dashboard.html', student=student, attendance_info=attendance_info)


@student_bp.route('/logout')
def student_logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('student.student_login'))


############################# ACCURACY EVALUATION ROUTE ####################################

testdata_bp = Blueprint('testdata', __name__, url_prefix='/testdata')
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
import os
import datetime
import base64
import csv
from .models import Student, Faculty, Subject, FacultySubject, AttendanceSession, AttendanceRecord, Admin
from . import db

testdata_bp = Blueprint('testdata', __name__, url_prefix='/testdata')

@testdata_bp.route('/add', methods=['GET', 'POST'])
def add_test_image():
    if request.method == 'POST':
        # Clean and normalize USN input
        usn = (request.form.get('usn') or '').strip().upper()
        image_data = request.form.get('image_data')

        if not usn or not image_data:
            flash('Please provide both USN and capture an image.')
            return redirect(url_for('testdata.add_test_image'))

        # Decode base64 image data safely
        try:
            header, encoded = image_data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
        except Exception as ex:
            flash(f'Invalid image data format: {ex}')
            return redirect(url_for('testdata.add_test_image'))

        # Set up paths for saving image and CSV
        save_dir = os.path.join(current_app.root_path, 'static', 'test_images')
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{usn}_{timestamp}.jpg"
        save_path = os.path.join(save_dir, filename)
        csv_path = os.path.join(current_app.root_path, 'static', 'test_labels.csv')

        # Save image and append entry to CSV file
        try:
            with open(save_path, 'wb') as f:
                f.write(img_bytes)
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"test_images/{filename}", usn])
            flash('Test image captured and saved successfully!')
        except Exception as e:
            flash(f'Error saving image or updating CSV: {str(e)}')

        return redirect(url_for('testdata.add_test_image'))

    # On GET, query all students for the USN dropdown datalist
    students = Student.query.all()
    return render_template('add_test_image.html', students=students)


@testdata_bp.route('/video_recognition_demo')
def video_recognition_demo():
    return render_template('testdata/video_recognition_demo.html')

def gen_cctv_video_frames(video_path, known_embeddings_dict, frame_skip=2, show_confidence=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Cannot open video!')
        return

    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        # Skip frames to reduce load and simulate throttling
        if frame_counter % frame_skip != 0:
            continue

        # Current timestamp overlay
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognize_faces_in_frame(rgb_frame, known_embeddings_dict)

        # Draw bounding boxes, labels, and optionally confidence
        for usn, box in results:
            if box is None:
                continue
            x1, y1, x2, y2 = map(int, box)

            # Custom bounding box colors (green if known, orange if unknown)
            color = (0, 255, 0) if usn else (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = str(usn) if usn else "Unknown"

            # Optionally display confidence (here we simulate with fixed value)
            if show_confidence:
                confidence = " (0.85)"  # Replace with actual if available
                label += confidence

            cv2.putText(frame, label, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Overlay timestamp top-left
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Encode to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

@testdata_bp.route('/video_demo_feed')
def video_demo_feed():
    video_path = os.path.join(current_app.static_folder, 'sample_video.mp4')

    students = Student.query.filter(Student.embedding.isnot(None)).all()
    known_embeddings_dict = {}
    for student in students:
        emb = load_embedding_from_db(student)
        if emb is not None:
            known_embeddings_dict[student.usn] = emb

    # Provide throttling by skipping every other frame
    return Response(
        gen_cctv_video_frames(video_path, known_embeddings_dict, frame_skip=2, show_confidence=True),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

