import os
import sys
import yaml
import pickle


from src.exception import CustomException

def load_yolo_label(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
        bboxes = []
        class_labels = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(float(parts[0]))  # 🔧 fixes '0.0' or '1.0'
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)
            class_labels.append(class_id)
        return bboxes, class_labels


def save_yolo_label(path, bboxes, class_labels):
    with open(path, "w") as f:
        for cls, bbox in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def generate_data_yaml(
    train_img_dir="data/annotated/train/images",
    val_img_dir="data/annotated/val/images",
    test_img_dir="data/annotated/test/images",
    class_names=["bottle"],
    yaml_path="data.yaml"
):
    data_config = {
        "train": train_img_dir,
        "val": val_img_dir,
        "test": test_img_dir,
        "nc": len(class_names),
        "names": class_names
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f)

