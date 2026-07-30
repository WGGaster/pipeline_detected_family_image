import json
import pandas as pd
from pathlib import Path
import sys
import random
from PIL import Image
from io import BytesIO
import requests

def download_image(img_http_path, path_save: Path):
  try:
    path_save.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(img_http_path, timeout=10)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert('RGB')
    img.save(path_save, format='JPEG')
    return True
  except Exception as e:
    print(f"Ошибка скачивания {img_http_path}: {e}")
    return False

def load_json_coco_2017(ann_path):
  with open(ann_path, 'r') as f:
    data_instance = json.load(f)
  return data_instance

def select_ids_img_of_category(data, category, ):
  ids_img_category = None
  for row in data:
    if row['category_id'] == category['id']:
      if ids_img_category is None:
        ids_img_category = [row['image_id']]
      elif ids_img_category[-1] != row['image_id']:
        ids_img_category.append(row['image_id'])
  return ids_img_category

def select_ids_img_not_in_category(data, category):
    target_id = category['id']
    has_category_ids = {row['image_id'] for row in data if row['category_id'] == target_id}
  
    all_image_ids = {row['image_id'] for row in data}
    return all_image_ids - has_category_ids

def download_images_with_http_path(list_imgs: list[tuple[str, str]], out_dir: str):
  count = 0
  for img in list_imgs:
      filename = img[0]
      http_path = img[1]
      out_path = out_dir / filename
      if not out_path.exists():
          if download_image(http_path, out_path):
              count += 1
              if count % 50 == 0:
                  print(f"Скачано {count} фото")
  print(f"Итого скачано: {count}")

def split_instances_coco_2017(data_instance, id_category):
  images = data_instance['images']
  annotations = data_instance['annotations']
  category  = data_instance['categories'][id_category]
  return images, annotations, category

def get_ids_img(annotations, category, without_category):
  if without_category:
    return select_ids_img_not_in_category(annotations, category)
  return select_ids_img_of_category(annotations, category)

def get_filenames_and_path_img(images, ids_img_category, size_sample):
  dict_id_image = {image['id']: (image['file_name'], image['coco_url']) for image in images}
  return random.sample([dict_id_image[id] for id in ids_img_category], size_sample)

def download_images_from_coco_2017(ann_path, out_dir, id_category, size_sample=1000, without_category=False):
  data_instance = load_json_coco_2017(ann_path)
  images, annotations, category = split_instances_coco_2017(data_instance, id_category)
  ids_img_category = get_ids_img(annotations, category, without_category)
  pairs_filename_path = get_filenames_and_path_img(images, ids_img_category, size_sample)
  download_images_with_http_path(pairs_filename_path, out_dir)

def download_images_has_not_person(ann_path, out_dir, is_val=False):
  ann_path = Path('../data/instance/instances_train2017.json')
  out_dir = Path('../data/coco2017_has_not_human')
  if is_val:
    ann_path = Path('../data/instance/instances_val2017.json')
  download_images_from_coco_2017(ann_path, out_dir, 0, without_category=True)

def download_images_has_person(is_val=False):
  ann_path = Path('../data/instance/instances_train2017.json')
  out_dir = Path('../data/coco2017_has_not_human')
  if is_val:
    ann_path = Path('../data/instance/instances_val2017.json')
  download_images_from_coco_2017(ann_path, out_dir, 0)
