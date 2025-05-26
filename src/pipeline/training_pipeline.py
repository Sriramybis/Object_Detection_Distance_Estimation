import sys
from src.logger import logging
from src.exception import CustomException

from src.components.image_train_test_split import ImageTrainTestSplitter, ImageSplitConfig
from src.components.augment import ImageAugmentor, ImageAugmentorConfig
from src.utils import generate_data_yaml
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig  

class TrainingPipeline:
    def __init__(self):
        # You can override config values here if needed
        self.split_config = ImageSplitConfig()
        self.augmentor_config = ImageAugmentorConfig()
        self.model_config = ModelTrainerConfig()

    def run(self):
        try:
            logging.info(" Starting the training pipeline...")

            # STEP 1: Train/Val/Test Split
            logging.info("Step 1: Splitting dataset")
            splitter = ImageTrainTestSplitter(self.split_config)
            train_dir, val_dir, test_dir = splitter.initiate_split()

            # STEP 2: Data Augmentation on Training Set
            logging.info("Step 2: Augmenting training data")
            augmentor = ImageAugmentor(self.augmentor_config)
            augmentor.initiate_image_augmentation()
            augmentor.consolidate_to_training(
                src_img_dir = "data/augmented/train/images",
                src_lbl_dir = "data/augmented/train/labels",
                dst_img_dir = train_dir + "/images",
                dst_lbl_dir = train_dir + "/labels")
            
            # STEP 3: Generating yaml for yolov5
            generate_data_yaml(
                train_img_dir="data/annotated/train/images",
                val_img_dir="data/annotated/val/images",
                test_img_dir="data/annotated/test/images",
                class_names=["bottle"],  # or pass more if needed
                yaml_path="data.yaml"
            )
                
            # STEP 4: Training
            logging.info("Step 3: Training the model")
            model_trainer = ModelTrainer(self.model_config)
            model_trainer.initiate_model_trainer()

            logging.info("Training pipeline completed successfully")

        except Exception as e:
            logging.error("Training pipeline failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
