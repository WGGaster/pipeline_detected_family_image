from pathlib import Path
import sys
from downloading_utils import download_images_from_coco_2017

def download_image_from_coco_2017_for_model_has_human(len_train_sample, len_test_sample):
  download_images_from_coco_2017(
      ann_path=Path('./data/instance/instances_train2017.json'),
      out_dir=Path('./data/train/coco2017_has_human_train'),
      id_category=0,
      size_sample=len_train_sample,
      without_category=False
  )

  download_images_from_coco_2017(
      ann_path=Path('./data/instance/instances_train2017.json'),
      out_dir=Path('./data/train/coco2017_has_not_human_train'),
      id_category=0,
      size_sample=len_train_sample,
      without_category=True
  )

  download_images_from_coco_2017(
      ann_path=Path('./data/instance/instances_val2017.json'),
      out_dir=Path('./data/val/coco2017_has_human_val'),
      id_category=0,
      size_sample=len_test_sample,
      without_category=False
  )

  download_images_from_coco_2017(
      ann_path=Path('./data/instance/instances_val2017.json'),
      out_dir=Path('./data/val/coco2017_has_not_human_val'),
      id_category=0,
      size_sample=len_test_sample,
      without_category=True
  )