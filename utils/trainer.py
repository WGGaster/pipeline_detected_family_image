from copy import deepcopy
import torch.nn as nn 
import torch
from tqdm import tqdm
from pathlib import Path

class Early_stopping_train:
  def __init__(self, deadline_num_epochs_train):
    self.deadline_num_epochs_train = deadline_num_epochs_train
    self.current_epoch = 0
    self.best_loss = None
    self.best_weights = None

  def try_save_best_loss(self, loss):
    if self.best_loss is None or loss < self.best_loss:
      self.best_loss = loss
      self.current_epoch = 0
      return True
    else:
      self.current_epoch += 1
      print('++++++++++++++++++++++++++++')
      print(f"Early Stopping: {self.current_epoch} / {self.deadline_num_epochs_train} эпох без улучшений. (Лучший лосс: {self.best_loss}, Текущий лосс: {loss})")
      print('++++++++++++++++++++++++++++')
      return False

  def isStopped(self):
    return self.current_epoch >= self.deadline_num_epochs_train

  def save_best_weights(self, model, loss, is_test=False):
    if is_test and self.try_save_best_loss(loss):
      self.best_weights = deepcopy(model.state_dict())

def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Test"):
        images = images.to(device)
        labels = labels.to(device).long()

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def get_y_true_pred(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in tqdm(loader, desc="Test"):
        images = images.to(device)
        labels = labels.to(device).long()
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels)
        y_pred.extend(preds)
    y_true = [y.cpu().item() for y in y_true]
    y_pred = [y.cpu().item() for y in y_pred]
    return y_true, y_pred

def train_model(model, train_loader, val_loader, path_save, criterion, optimizer, scheduler, num_epochs=15, 
                deadline_num_epochs_train=5, device='cpu'):
  path_save = Path(path_save)
  early_stopping = Early_stopping_train(deadline_num_epochs_train)
  train_loss_list, train_acc_list = [], []
  val_loss_list, val_acc_list = [], []


  for epoch in range(num_epochs):
      print(f"\nEpoch {epoch+1}/{num_epochs}")
      train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, device)
      train_loss_list.append(train_loss)
      train_acc_list.append(train_acc)
      val_loss, val_acc = validate(val_loader, model, criterion, device)
      val_loss_list.append(val_loss)
      val_acc_list.append(val_acc)
      scheduler.step()
      early_stopping.save_best_weights(model, val_loss, is_test=True)
      print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
      print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
      print('--------------------')
      if early_stopping.isStopped():
        print("Сработала рання остановка!")
        break
      print('--------------------')
  
  path_save.parent.mkdir(parents=True, exist_ok=True)
  model.load_state_dict(early_stopping.best_weights)
  torch.save(model.state_dict(), path_save)
  return {
      'train loss': train_loss_list,
      'train accuracy': train_acc_list,
      'val loss': val_loss_list,
      'val accuracy': val_acc_list
  }
