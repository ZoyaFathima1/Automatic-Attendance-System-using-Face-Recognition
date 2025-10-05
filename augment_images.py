import os
import cv2
import glob
import csv
import albumentations as A

# Directories - adjust after splitting
TRAIN_DIR = 'instance/train'                  # train split folder (after splitting)
AUG_DIR = 'instance_augmented/train'          # augmented output for train images only
LABEL_CSV = os.path.join(AUG_DIR, 'augmented_labels.csv')


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.6),
    A.OneOf([
        A.CoarseDropout(max_holes=1, max_height=40, max_width=40, 
                        min_holes=1, min_height=20, min_width=20, fill_value=0, p=0.5),
        A.Cutout(num_holes=1, max_h_size=40, max_w_size=40, fill_value=0, p=0.5),
    ], p=0.5),
])


def augment_and_save_image(image_path, output_folder, prefix, count=5):
    os.makedirs(output_folder, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return []

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    saved_files = []

    orig_out_path = os.path.join(output_folder, f"{prefix}_{base_name}_orig.jpg")
    cv2.imwrite(orig_out_path, img)
    saved_files.append(orig_out_path)

    for i in range(count):
        augmented = transform(image=img)['image']
        out_path = os.path.join(output_folder, f"{prefix}_{base_name}_aug{i + 1}.jpg")
        cv2.imwrite(out_path, augmented)
        saved_files.append(out_path)

    return saved_files


def augment_dataset():
    all_rows = []
    for usn_folder in os.listdir(TRAIN_DIR):
        usn_path = os.path.join(TRAIN_DIR, usn_folder)
        if not os.path.isdir(usn_path):
            continue
        aug_folder = os.path.join(AUG_DIR, usn_folder)
        image_paths = glob.glob(os.path.join(usn_path, '*.jpg')) + \
                      glob.glob(os.path.join(usn_path, '*.jpeg')) + \
                      glob.glob(os.path.join(usn_path, '*.png'))

        print(f"Augmenting {len(image_paths)} images for student {usn_folder}...")
        for img_path in image_paths:
            saved_files = augment_and_save_image(img_path, aug_folder, usn_folder)
            for f in saved_files:
                rel_path = os.path.relpath(f, AUG_DIR).replace('\\', '/')
                all_rows.append([rel_path, usn_folder.upper()])

    os.makedirs(AUG_DIR, exist_ok=True)
    with open(LABEL_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'usn'])
        writer.writerows(all_rows)

    print(f"Augmentation complete. Labels saved to {LABEL_CSV}")


if __name__ == "__main__":
    augment_dataset()
