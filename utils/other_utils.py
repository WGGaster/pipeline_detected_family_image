from torchvision import models
import torch.nn as nn
import torch
import json
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

def get_model_resnet18(num_classes=2, pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes)
    )
    return model

def view_classification_report(y_true, y_pred, target_names):
  report = classification_report(y_true, y_pred, target_names=target_names)
  print(report)

def load_model(path_model):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = get_model_resnet18(pretrained=False)
  model.to(device)
  state_dict = torch.load(Path(path_model), map_location=device, weights_only=True)
  model.load_state_dict(state_dict, strict=True)
  return model


def save_json(data, path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
  
  print(f"JSON успешно сохранен по пути: {path}")
