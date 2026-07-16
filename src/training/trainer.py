import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class Trainer:
    """
    Manages the training epoch loop, evaluation, learning rate schedules,
    and Mixed Precision optimization.
    """
    def __init__(self, model, optimizer, criterion, scaler, device, accumulation_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.device = device
        self.accumulation_steps = accumulation_steps

    def train_epoch(self, loader):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Mixed Precision autocasting
            with torch.amp.autocast(device_type="cuda" if "cuda" in self.device.type else "cpu", dtype=torch.float16):
                logits = self.model(inputs)
                raw_loss = self.criterion(logits, targets)
                loss = raw_loss / self.accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            # Gradient Accumulation Boundary
            is_accumulation_boundary = (batch_idx + 1) % self.accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(loader)
            
            if is_accumulation_boundary or is_last_batch:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
            total_loss += raw_loss.item() * inputs.size(0)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        return total_loss / total, correct / total

    @staticmethod
    def get_cosine_lr(epoch, total_epochs, max_lr, cycles=4):
        """
        Cosine learning rate scheduler.
        """
        epochs_per_cycle = math.ceil(total_epochs / cycles)
        cycle_epoch = epoch % epochs_per_cycle
        lr = max_lr / 2.0 * (math.cos(math.pi * cycle_epoch / epochs_per_cycle) + 1.0)
        return max(lr, 1e-6)

    def evaluate_clips(self, models, records, cached_waveforms, frame_length=8000):
        """
        Evaluates whole 4-second audio clips using the SUM rule on all 15 overlapping frames
        retrieved directly from RAM. Supports ensembling.
        """
        for m in models:
            m.eval()
            
        # Group records by audio path
        clips = defaultdict(list)
        for r in records:
            clips[r["path"]].append(r)
            
        correct = 0
        total = len(clips)
        
        # Pre-generate frame offsets
        offsets = [i * 4000 for i in range(15)] # 50% overlap of 8000 frames
        
        with torch.no_grad():
            for path, frames in clips.items():
                label = frames[0]["label"]
                waveform_np = cached_waveforms[path]
                waveform = torch.from_numpy(waveform_np)
                    
                # Extract all 15 frames
                batch_frames = []
                for offset in offsets:
                    frame = waveform[:, offset:offset + frame_length]
                    if frame.shape[-1] < frame_length:
                        frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode='constant')
                    batch_frames.append(frame)
                    
                # Shape: (15, 1, 8000)
                batch_tensor = torch.stack(batch_frames).to(self.device)
                
                # Forward pass through all ensembled models
                sum_probs = torch.zeros((15, 10), device=self.device)
                for m in models:
                    with torch.amp.autocast(device_type="cuda" if "cuda" in self.device.type else "cpu", dtype=torch.float16):
                        logits = m(batch_tensor)
                        probs = F.softmax(logits, dim=-1)
                        sum_probs += probs
                
                sum_probs /= len(models)
                
                # SUM rule: aggregate frame predictions
                clip_prob = torch.sum(sum_probs, dim=0) # (10,)
                predicted_class = torch.argmax(clip_prob).item()
                
                if predicted_class == label:
                    correct += 1
                    
        return correct / total
