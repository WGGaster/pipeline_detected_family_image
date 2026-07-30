import kagglehub
from pathlib import Path
import random
import shutil

def get_image_files(src_dir: Path) -> list[Path]:
    if not src_dir.exists():
        return []
    return [
        f for f in src_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

def download_images_blur_dataset(train_count: int = 250, val_count: int = 100, seed: int = 42):
    # 1. Скачиваем датасет
    root_dir = Path(kagglehub.dataset_download("kwentar/blur-dataset"))
    print(f"Датасет скачан в: {root_dir}")

    # 2. Собираем файлы из реальных папок датасета
    sharp_dir = root_dir / "sharp"
    defocus_dir = root_dir / "defocused_blurred"
    motion_dir = root_dir / "motion_blurred"

    sharp_files = get_image_files(sharp_dir)
    defocus_files = get_image_files(defocus_dir)
    motion_files = get_image_files(motion_dir)

    if len(sharp_files) < (train_count + val_count):
        raise RuntimeError(
            f"Недостаточно изображений в классе 'sharp': нужно {train_count + val_count}, "
            f"а найдено {len(sharp_files)}."
        )
    if (len(defocus_files) + len(motion_files)) < (train_count + val_count):
        raise RuntimeError(
            f"Недостаточно изображений в классе 'blur': нужно {train_count + val_count}, "
            f"а найдено {len(defocus_files) + len(motion_files)}."
        )

    # Объединяем оба типа размытия в один класс
    blur_files = defocus_files + motion_files

    # 3. Фиксированный сплит: первые n — train, следующие m — val (после перемешивания)
    random.seed(seed)
    random.shuffle(sharp_files)
    random.shuffle(blur_files)

    sharp_train = sharp_files[:train_count]
    sharp_val = sharp_files[train_count:train_count + val_count]

    blur_train = blur_files[:train_count]
    blur_val = blur_files[train_count:train_count + val_count]

    # 4. Настраиваем пути
    base_path = Path("./data")

    # Папки train
    train_blur_folder = base_path / "train" / "blur_dataset" / 'blur'
    train_sharp_folder = base_path / "train" / "blur_dataset" / 'sharp'

    # Папки val
    val_blur_folder = base_path / "val" / "blur_dataset" / 'blur'
    val_sharp_folder = base_path / "val" / "blur_dataset" / 'sharp'

    # Создаём папки, если их нет
    train_blur_folder.mkdir(parents=True, exist_ok=True)
    train_sharp_folder.mkdir(parents=True, exist_ok=True)
    val_blur_folder.mkdir(parents=True, exist_ok=True)
    val_sharp_folder.mkdir(parents=True, exist_ok=True)

    print("Целевые папки:")
    print(f"  Train blur: {train_blur_folder}")
    print(f"  Train sharp: {train_sharp_folder}")
    print(f"  Val blur: {val_blur_folder}")
    print(f"  Val sharp: {val_sharp_folder}")

    # 5. Копируем файлы
    def copy_files(files: list[Path], dst_dir: Path) -> int:
        count = 0
        for f in files:
            dst_file = dst_dir / f.name
            if not dst_file.exists():
                shutil.copy(f, dst_file)
                count += 1
        return count

    c1 = copy_files(blur_train, train_blur_folder)
    c2 = copy_files(sharp_train, train_sharp_folder)
    c3 = copy_files(blur_val, val_blur_folder)
    c4 = copy_files(sharp_val, val_sharp_folder)

    print(f"Скопировано в train/blur: {c1}")
    print(f"Скопировано в train/sharp: {c2}")
    print(f"Скопировано в val/blur: {c3}")
    print(f"Скопировано в val/sharp: {c4}")
