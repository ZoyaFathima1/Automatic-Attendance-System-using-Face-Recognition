import os
import traceback
from app import create_app, db
from app.models import Student
from app.utils import compute_and_store_embeddings

app = create_app()
app.app_context().push()

# Updated path to point to augmented training images folder
AUGMENTED_DIR = os.path.abspath('instance_augmented/train')

def generate_all_embeddings(auto_clear=True):
    students = Student.query.all()
    for student in students:
        usn = student.usn.strip().upper()
        folder = os.path.join(AUGMENTED_DIR, usn)

        if not os.path.exists(folder):
            print(f"[WARNING] No augmented folder found for {usn}, skipping.")
            continue

        if student.embedding and not auto_clear:
            print(f"[INFO] Skipping {usn}, embedding already exists.")
            continue

        try:
            print(f"[INFO] Processing embeddings for {usn} from augmented training images...")
            ok = compute_and_store_embeddings(usn, folder, Student, db, auto_clear=auto_clear, min_prob=0.95)
            if ok:
                print(f"[SUCCESS] Saved embeddings for {usn}")
            else:
                print(f"[WARNING] No valid faces found for {usn}")
        except Exception as e:
            print(f"[ERROR] Exception processing {usn}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    generate_all_embeddings(auto_clear=True)
