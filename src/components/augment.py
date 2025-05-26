import os
import sys
import cv2
import shutil
import albumentations as A

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import load_yolo_label, save_yolo_label



@dataclass
class ImageAugmentorConfig:
    input_img_dir: str = "data/annotated/train/images"
    input_lbl_dir: str = "data/annotated/train/labels"
    output_img_dir: str = "data/augmented/train/images"
    output_lbl_dir: str = "data/augmented/train/labels"
    aug_per_image: int = 2


class ImageAugmentor:
    def __init__(self, config: ImageAugmentorConfig = ImageAugmentorConfig()):
        self.config = config
        os.makedirs(self.config.output_img_dir, exist_ok=True)
        os.makedirs(self.config.output_lbl_dir, exist_ok=True)

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Rotate(limit=10, p=0.5),
            A.Blur(blur_limit=3, p=0.2)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def initiate_image_augmentation(self):
        logging.info("Entered image augmentation component")
        try:
            image_files = [f for f in os.listdir(self.config.input_img_dir) if f.endswith(".jpg")]
            logging.info(f"Found {len(image_files)} images for augmentation")

            for img_name in image_files:
                base_name = os.path.splitext(img_name)[0]
                img_path = os.path.join(self.config.input_img_dir, img_name)
                lbl_path = os.path.join(self.config.input_lbl_dir, base_name + ".txt")

                if not os.path.exists(lbl_path):
                    logging.warning(f"No label found for {img_name}, skipping.")
                    continue

                image = cv2.imread(img_path)
                bboxes, class_labels = load_yolo_label(lbl_path)

                # Copy original
                shutil.copy2(img_path, os.path.join(self.config.output_img_dir, img_name))
                shutil.copy2(lbl_path, os.path.join(self.config.output_lbl_dir, base_name + ".txt"))

                for i in range(self.config.aug_per_image):
                    transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_img = transformed["image"]
                    aug_bboxes = transformed["bboxes"]
                    aug_labels = transformed["class_labels"]

                    if not aug_bboxes:
                        logging.warning(f"Augmented image {base_name}_aug{i} has no valid bboxes, skipping.")
                        continue

                    aug_name = f"{base_name}_aug{i}"
                    cv2.imwrite(os.path.join(self.config.output_img_dir, aug_name + ".jpg"), aug_img)
                    save_yolo_label(os.path.join(self.config.output_lbl_dir, aug_name + ".txt"), aug_bboxes, aug_labels)

            logging.info("Image augmentation completed successfully")

            return self.config.output_img_dir, self.config.output_lbl_dir
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def consolidate_to_training(self, train_img_dir, train_lbl_dir):
        """
        Copy all images and labels from the augmented output directory
        into the given training image and label directories.
        """
        try:
            for img_file in os.listdir(self.config.output_img_dir):
                shutil.copy2(
                    os.path.join(self.config.output_img_dir, img_file),
                    os.path.join(train_img_dir, img_file)
                )

            for lbl_file in os.listdir(self.config.output_lbl_dir):
                shutil.copy2(
                    os.path.join(self.config.output_lbl_dir, lbl_file),
                    os.path.join(train_lbl_dir, lbl_file)
                )

            logging.info(" Consolidated augmented data into training folder")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    augmentor = ImageAugmentor()
    aug_img_path, aug_lbl_path = augmentor.initiate_image_augmentation()
    augmentor.consolidate_to_training(aug_img_path,aug_lbl_path)

    # Optional consolidation step (uncomment if you want to merge into training set)
    # for file in os.listdir(aug_img_path):
    #     shutil.copy2(os.path.join(aug_img_path, file), os.path.join("data/annotated/train/images", file))
    # for file in os.listdir(aug_lbl_path):
    #     shutil.copy2(os.path.join(aug_lbl_path, file), os.path.join("data/annotated/train/labels", file))
