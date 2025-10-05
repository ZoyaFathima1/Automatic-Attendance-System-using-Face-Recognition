import os
import random
import shutil

ORIG_ROOT = 'instance/student_images'
TRAIN_ROOT = 'instance/train'
VAL_ROOT = 'instance/val'

for usn_folder in os.listdir(ORIG_ROOT):
    usn_path = os.path.join(ORIG_ROOT, usn_folder)
    if not os.path.isdir(usn_path):
        continue
    images = [f for f in os.listdir(usn_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(images)
    train_imgs = images[:3]  # 3 for train
    val_imgs = images[3:]    # remaining 2 for val

    for img in train_imgs:
        src = os.path.join(usn_path, img)
        dst = os.path.join(TRAIN_ROOT, usn_folder)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

    for img in val_imgs:
        src = os.path.join(usn_path, img)
        dst = os.path.join(VAL_ROOT, usn_folder)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)
