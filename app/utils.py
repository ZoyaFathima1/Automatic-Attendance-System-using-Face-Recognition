import os
import numpy as np
from PIL import Image
import logging
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup device for GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models (singletons): aligned crops + FaceNet backbone
mtcnn = MTCNN(keep_all=True, image_size=160, margin=20, device=device)  # aligned crops expected by FaceNet
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)   # 512-d embeddings

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v) + 1e-10
    return v / n

def compute_and_store_embeddings(usn, student_folder, Student, db, auto_clear=False, min_prob=0.95):
    """
    Build a single normalized template per student by averaging multiple normalized embeddings
    from aligned, high-probability MTCNN crops, then re-normalize the mean before saving.
    """
    per_face_embs = []
    total_images, total_faces, failed_images = 0, 0, []

    for filename in os.listdir(student_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        total_images += 1
        img_path = os.path.join(student_folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            failed_images.append(filename)
            continue

        # Get aligned crops + confidences
        faces, probs = mtcnn(img, return_prob=True)
        if faces is None or len(faces) == 0:
            failed_images.append(filename)
            continue

        for face, p in zip(faces, probs):
            if p is None or p < min_prob:
                continue
            with torch.no_grad():
                face = face.to(device)
                emb = resnet(face.unsqueeze(0)).cpu().numpy()[0]
            emb = l2_normalize(emb)
            per_face_embs.append(emb)
            total_faces += 1

    if not per_face_embs:
        logger.warning(f"No face embeddings found for {usn}. Failed images: {failed_images}")
        return False

    mean_emb = l2_normalize(np.mean(per_face_embs, axis=0))

    student = Student.query.filter_by(usn=usn).first()
    if not student:
        logger.warning(f"Student not found for usn={usn}")
        return False

    if auto_clear:
        student.embedding = None
        db.session.commit()

    student.embedding = mean_emb.astype(np.float32).tobytes()
    db.session.commit()

    logger.info(f"Student {usn}: Images={total_images}, Faces={total_faces}, Failed={len(failed_images)}")
    if failed_images:
        logger.info(f"Failed images (no/low-prob faces): {failed_images}")
    return True


def load_embedding_from_db(student):
    if student.embedding:
        vec = np.frombuffer(student.embedding, dtype=np.float32)
        return vec
    return None


def compare_embeddings_cosine(emb1, emb2):
    # emb1/emb2 are assumed normalized
    return float(np.dot(emb1, emb2))


def precompute_embeddings_for_images(image_paths, min_prob=0.95):
    """ Compute and cache embeddings for a list of image paths. """
    embeddings = {}
    for img_path in image_paths:
        if not os.path.isfile(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not open image {img_path}: {e}")
            continue
        faces, probs = mtcnn(img, return_prob=True)
        if faces is None or len(faces) == 0:
            continue
        valid_embs = []
        for face, p in zip(faces, probs):
            if p is None or p < min_prob:
                continue
            with torch.no_grad():
                face = face.to(device)
                emb = resnet(face.unsqueeze(0)).cpu().numpy()[0]
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            valid_embs.append(emb)
        if valid_embs:
            # Average multiple face embeddings
            embeddings[img_path] = np.mean(valid_embs, axis=0)
    return embeddings


def recognize_faces_with_cached_embeddings(frame_embeddings, known_embeddings_dict, threshold=0.7):
    """
    Recognize faces by comparing precomputed embeddings directly instead of re-running models.
    - frame_embeddings: list of embeddings numpy arrays representing faces in the frame.
    - Returns list of predicted (usn, None) since boxes aren't included here.
    """
    norm_gallery = {k: (v / (np.linalg.norm(v) + 1e-10)) for k, v in known_embeddings_dict.items()}
    results = []

    for emb in frame_embeddings:
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        best_usn, best_sim = None, -1.0
        for usn, g in norm_gallery.items():
            sim = float(np.dot(emb_norm, g))
            if sim > best_sim:
                best_sim, best_usn = sim, usn
        pred = best_usn if best_sim >= threshold else None
        results.append((pred, None))
    return results


def recognize_faces_in_frame(frame_bgr_or_rgb, known_embeddings_dict, threshold=0.7, min_prob=0.90):
    """
    Legacy function to detect faces and compute embeddings one by one.
    Use precompute_embeddings_for_images + recognize_faces_with_cached_embeddings for better performance.
    """
    # Ensure gallery is normalized once
    norm_gallery = {k: (v / (np.linalg.norm(v) + 1e-10)) for k, v in known_embeddings_dict.items()}

    # Convert to PIL RGB
    if frame_bgr_or_rgb.ndim == 3 and frame_bgr_or_rgb.shape[2] == 3:
        img_pil = Image.fromarray(frame_bgr_or_rgb.astype('uint8'))
    else:
        raise ValueError("Frame must be HxWx3 array")

    # Get aligned crops + probabilities
    faces, probs = mtcnn(img_pil, return_prob=True)
    # Get boxes for visualization
    boxes, _ = mtcnn.detect(img_pil)

    results = []
    if faces is None or len(faces) == 0:
        return results

    for idx, face in enumerate(faces):
        p = probs[idx] if probs is not None and idx < len(probs) else None
        if p is None or p < min_prob:
            results.append((None, boxes[idx] if boxes is not None and idx < len(boxes) else None))
            continue

        with torch.no_grad():
            face = face.to(device)
            emb = resnet(face.unsqueeze(0)).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-10)

        # Argmax cosine over gallery
        best_usn, best_sim = None, -1.0
        for usn, g in norm_gallery.items():
            sim = float(np.dot(emb, g))
            if sim > best_sim:
                best_sim, best_usn = sim, usn

        pred = best_usn if best_sim >= threshold else None
        box = boxes[idx] if boxes is not None and idx < len(boxes) else None
        results.append((pred, box))
    return results
