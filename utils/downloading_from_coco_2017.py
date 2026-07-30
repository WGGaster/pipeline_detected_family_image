import os
import shutil
from pathlib import Path
import kagglehub
import subprocess
import zipfile
from downloading_utils import download_images_from_coco_2017

import subprocess
import zipfile
import shutil
from pathlib import Path

def check_and_download_coco_2017_annotations():
    train_path = Path('./data/instance/instances_train2017.json')
    val_path = Path('./data/instance/instances_val2017.json')
    
    
    if train_path.exists() and val_path.exists():
        return
        
    print("Текстовые файлы аннотаций не найдены. Распаковываю локальный архив...")
    
   
    zip_path = Path('./data/instance/instance-20260730T091317Z-1-001.zip')
    
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Критическая ошибка: Архив {zip_path} не найден! "
            f"Пожалуйста, убедитесь, что zip-файл лежит в папке ./data/instance/"
        )
        
    extract_tmp_dir = Path('./data/instance/tmp_extracted')
    extract_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_tmp_dir)
        
  
    extracted_train = extract_tmp_dir / "instance" / "instances_train2017.json"
    extracted_val = extract_tmp_dir / "instance" / "instances_val2017.json"
    
    if extracted_train.exists() and extracted_val.exists():
        shutil.move(str(extracted_train), str(train_path))
        shutil.move(str(extracted_val), str(val_path))
        print("🎉 Аннотации успешно извлечены напрямую в data/instance/!")
    else:
        raise FileNotFoundError(
            f"Не удалось найти файлы внутри архива. Проверьте структуру.\n"
            f"Ожидалось: {extracted_train}"
        )
        
    shutil.rmtree(extract_tmp_dir, ignore_errors=True)

def download_image_from_coco_2017_for_model_has_human(len_train_sample, len_val_sample):
  check_and_download_coco_2017_annotations()
  download_images_from_coco_2017(
      ann_path=Path('./data/instance/instances_train2017.json'),
      out_dir=Path('./data/train/coco2017/has_human'),
      id_category=0,
      size_sample=len_train_sample,
      without_category=False
  )

  download_images_from_coco_2017(
      ann_path=Path('./data/instance/instances_train2017.json'),
      out_dir=Path('./data/train/coco2017/has_not_human'),
      id_category=0,
      size_sample=len_train_sample,
      without_category=True
  )

  download_images_from_coco_2017(
      ann_path=Path('./data/instance/instances_val2017.json'),
      out_dir=Path('./data/val/coco2017/has_human'),
      id_category=0,
      size_sample=len_val_sample,
      without_category=False
  )

  download_images_from_coco_2017(
      ann_path=Path('./data/instance/instances_val2017.json'),
      out_dir=Path('./data/val/coco2017/has_not_human'),
      id_category=0,
      size_sample=len_val_sample,
      without_category=True
  )