import sys
from src.logger import logging
from src.exception import CustomException

from src.components.image_train_test_split import ImageTrainTestSplitter, ImageSplitConfig
from src.components.augment import ImageAugmentor, ImageAugmentorConfig
# from src.components.train import ModelTrainer, ModelTrainerConfig  # if applicable

class TrainingPipeline:
    def __init__(self):
        # You can override config values here if needed
        self.split_config = ImageSplitConfig()
        self.augmentor_config = ImageAugmentorConfig()

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
            aug_img_path, aug_lbl_path = augmentor.initiate_image_augmentation()
            augmentor.consolidate_to_training(train_img_dir=train_dir + "/images",
                                              train_lbl_dir=train_dir + "/labels")

            # STEP 3: Training (optional)
            # logging.info("Step 3: Training the model")
            # model_trainer = ModelTrainer(ModelTrainerConfig())
            # model_trainer.initiate_model_trainer()

            logging.info("Training pipeline completed successfully")

        except Exception as e:
            logging.error("Training pipeline failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
