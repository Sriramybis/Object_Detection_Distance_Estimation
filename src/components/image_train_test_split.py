# Make sense.ai used for annotating the images

import os
import sys
import shutil
import random

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException



@dataclass
class ImageSplitConfig:
    raw_img_dir: str = "data/raw/images"
    annotation_dir: str = "data/raw/annotations"
    output_base: str = "data/annotated"
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1


class ImageTrainTestSplitter:
    def __init__(self, config: ImageSplitConfig = ImageSplitConfig()):
        self.config = config

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(config.output_base, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(config.output_base, split, 'labels'), exist_ok=True)

    def copy_files(self, file_list, split_name):
        for img_file in file_list:
            label_file = os.path.splitext(img_file)[0] + ".txt"

            src_img = os.path.join(self.config.raw_img_dir, img_file)
            src_lbl = os.path.join(self.config.annotation_dir, label_file)

            dst_img = os.path.join(self.config.output_base, split_name, 'images', img_file)
            dst_lbl = os.path.join(self.config.output_base, split_name, 'labels', label_file)

            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                logging.warning(f"Label not found for {img_file}, skipping label.")

    def initiate_split(self):
        try:
            logging.info("Started image train-val-test split process")

            image_files = [f for f in os.listdir(self.config.raw_img_dir) if f.endswith('.jpg')]
            random.shuffle(image_files)

            total = len(image_files)
            train_end = int(total * self.config.train_ratio)
            val_end = train_end + int(total * self.config.val_ratio)

            train_files = image_files[:train_end]
            val_files = image_files[train_end:val_end]
            test_files = image_files[val_end:]

            self.copy_files(train_files, 'train')
            self.copy_files(val_files, 'val')
            self.copy_files(test_files, 'test')

            logging.info("✅ Image split completed successfully.")
            return (
                os.path.join(self.config.output_base, 'train'),
                os.path.join(self.config.output_base, 'val'),
                os.path.join(self.config.output_base, 'test'),
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    splitter = ImageTrainTestSplitter()
    train_dir, val_dir, test_dir = splitter.initiate_split()
