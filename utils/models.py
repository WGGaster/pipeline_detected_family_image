import torch
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
from dataset import ConvDataset
import json
from other_utils import save_json, load_model
from tqdm import tqdm
import shutil


def download_models_MTCNN():
    weights_dir = Path('models/face_detector')
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("Скачивание предобученной архитектуры MTCNN...")
    mtcnn_downloader = MTCNN(keep_all=True)
    torch.save(mtcnn_downloader.pnet.state_dict(), weights_dir / 'pnet_weights.pt')
    torch.save(mtcnn_downloader.rnet.state_dict(), weights_dir / 'rnet_weights.pt')
    torch.save(mtcnn_downloader.onet.state_dict(), weights_dir / 'onet_weights.pt')
    print("Веса успешно сохранены в локальные файлы .pt в папку models/face_detector/!")


class Label:
    def __init__(self):
        with open('labels/label_has_human.json', 'r', encoding='utf-8') as f:
            self.labels_has_human = json.load(f)
            
        with open('labels/label_face.json', 'r', encoding='utf-8') as f:
            self.labels_face = json.load(f)
            
        with open('labels/label_blur.json', 'r', encoding='utf-8') as f:
            self.labels_blur = json.load(f)

class CustomFaceDetectionModel(torch.nn.Module):
    def __init__(
        self,
        pnet_path=Path('models/face_detector/pnet_weights.pt'),
        rnet_path=Path('models/face_detector/rnet_weights.pt'),
        onet_path=Path('models/face_detector/onet_weights.pt'),
        label_path=Path('labels/label_face.json'),
        device=None
    ):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.detector = MTCNN(keep_all=True, device=self.device, post_process=False)
        
        self.detector.pnet.load_state_dict(torch.load(pnet_path, map_location=self.device))
        self.detector.rnet.load_state_dict(torch.load(rnet_path, map_location=self.device))
        self.detector.onet.load_state_dict(torch.load(onet_path, map_location=self.device))
        
        self.dict_class_label = {'defect': 0, 'correct': 1}
        
        label_path = Path(label_path)
        if not label_path.exists():
            label_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(self.dict_class_label, label_path)
        self.eval()

    def forward(self, image_path):
        if not Path(image_path).exists():
            return self.dict_class_label['defect']
            
        try:
            img = Image.open(Path(image_path)).convert('RGB')
        except Exception:
            return self.dict_class_label['defect'] 
        
        with torch.no_grad():
            boxes, probs, landmarks = self.detector.detect(img, landmarks=True)
            
        if boxes is None:
            return self.dict_class_label['defect']
            
        for i, face_landmarks in enumerate(landmarks):
            if probs[i] < 0.90: 
                continue
                
            left_eye = face_landmarks[0]
            right_eye = face_landmarks[1]
            nose = face_landmarks[2]
            left_mouth = face_landmarks[3]
            right_mouth = face_landmarks[4]
            
            dist_l = abs(left_eye[0] - nose[0])
            dist_r = abs(right_eye[0] - nose[0])
            symmetry = max(dist_l, dist_r) / (min(dist_l, dist_r) + 1e-6)
            
            if symmetry > 2.5:
                return self.dict_class_label['defect']
                
            eye_slope = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0] + 1e-6)
            mouth_slope = (right_mouth[1] - left_mouth[1]) / (right_mouth[0] - left_mouth[0] + 1e-6)
            
            if abs(eye_slope - mouth_slope) > 0.35:
                return self.dict_class_label['defect']
                
        return self.dict_class_label['correct']


class FamilyImageFilterPipeline:
    def __init__(self, model_people, model_blur, model_face, base_result_dir="result", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_people = model_people.to(self.device).eval()
        self.model_blur = model_blur.to(self.device).eval()
        self.model_face = model_face.to(self.device).eval()
        self.transform = ConvDataset._default_transform()
        
        self.output_dir = self._generate_next_result_dir(base_result_dir)
        self.correct_dir = self.output_dir / "correct"
        self.defect_dir = self.output_dir / "defect"
        
        self.correct_dir.mkdir(parents=True, exist_ok=True)
        self.defect_dir.mkdir(parents=True, exist_ok=True)
        
        self.labels = {'defect': 0, 'correct': 1}
        self.label_obj = Label()
        self.correct_counter = 0
        self.defect_counter = 0
        
        print(f"Пайплайн инициализирован. Результаты этого прогона будут сохранены в: {self.output_dir}")

    def _generate_next_result_dir(self, base_name):
        counter = 0
        while True:
            folder_path = Path(f"result/{base_name}_{counter}")
            if not folder_path.exists():
                return folder_path
            counter += 1

    def pipeline_filter_family_image(self, image_path):
        img_path = Path(image_path)
        img_ext = img_path.suffix.lower() 
        
        if not img_path.exists():
            return self.labels['defect']
        try:
            raw_img = Image.open(img_path).convert('RGB')
            tensor_img = self.transform(raw_img).unsqueeze(0).to(self.device)
        except Exception:
            new_name = f"defect_{self.defect_counter}{img_ext}"
            shutil.copy(img_path, self.defect_dir / new_name)
            self.defect_counter += 1
            return self.labels['defect']

        with torch.no_grad():
            people_out = self.model_people(tensor_img)
            people_pred = torch.argmax(people_out, dim=1).item()
            
            if people_pred == self.label_obj.labels_has_human['has_not_human']:
                new_name = f"defect_{self.defect_counter}{img_ext}"
                shutil.copy(img_path, self.defect_dir / new_name)
                self.defect_counter += 1
                return self.labels['defect']
            
            blur_out = self.model_blur(tensor_img)
            blur_pred = torch.argmax(blur_out, dim=1).item()
            
            if blur_pred == self.label_obj.labels_blur['blur']:
                new_name = f"defect_{self.defect_counter}{img_ext}"
                shutil.copy(img_path, self.defect_dir / new_name)
                self.defect_counter += 1
                return self.labels['defect']

        face_pred = self.model_face(str(img_path))
        
        if face_pred == self.label_obj.labels_face['defect']:
            new_name = f"defect_{self.defect_counter}{img_ext}"
            shutil.copy(img_path, self.defect_dir / new_name)
            self.defect_counter += 1
            return self.labels['defect']
            
        new_name = f"correct_{self.correct_counter}{img_ext}"
        shutil.copy(img_path, self.correct_dir / new_name)
        self.correct_counter += 1
        return self.labels['correct']


def evaluate_pipeline_on_samples(pipeline_instance, samples_list):
    y_true = []
    y_pred = []
    
    print(f"Анализ тестового датасета напрямую через samples ({len(samples_list)} изображений)...")
    
    for img_path, true_label in tqdm(samples_list):
        predicted_label = pipeline_instance.pipeline_filter_family_image(img_path)
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
        
    return y_true, y_pred

def create_FamilyImageFilterPipeline(model_human_path, model_blur_path):
    face_model = CustomFaceDetectionModel()
    model_human = load_model(Path(model_human_path))
    model_blur = load_model(Path(model_blur_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = FamilyImageFilterPipeline(model_people=model_human, model_blur=model_blur, model_face=face_model, device=device)
    return pipeline