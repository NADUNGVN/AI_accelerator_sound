import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class Trainer:
    """
    Manages the training epoch loop, evaluation, learning rate schedules,
    and optional mixed precision optimization.
    """
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scaler,
        device,
        accumulation_steps=1,
        use_amp=True,
        gradient_clip=5.0,
        input_transform=None,
        mixup_cfg=None,
        ema=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.input_transform = input_transform
        self.mixup_cfg = mixup_cfg or {}
        self.ema = ema

    def transform_inputs(self, inputs):
        if self.input_transform is None:
            return inputs
        return self.input_transform(inputs.float())

    def train_epoch(self, loader):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            inputs = self.transform_inputs(inputs)
            mixup_enabled = bool(self.mixup_cfg.get("enabled", False)) and inputs.size(0) > 1
            if mixup_enabled and random.random() < float(self.mixup_cfg.get("prob", 1.0)):
                alpha = float(self.mixup_cfg.get("alpha", 0.2))
                lam = random.betavariate(alpha, alpha) if alpha > 0.0 else 1.0
                index = torch.randperm(inputs.size(0), device=self.device)
                mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
                targets_a = targets
                targets_b = targets[index]
            else:
                lam = 1.0
                mixed_inputs = inputs
                targets_a = targets
                targets_b = targets
            
            with torch.amp.autocast(
                device_type="cuda" if "cuda" in self.device.type else "cpu",
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                logits = self.model(mixed_inputs)
                if mixup_enabled and lam < 1.0:
                    raw_loss = lam * self.criterion(logits, targets_a) + (1.0 - lam) * self.criterion(logits, targets_b)
                else:
                    raw_loss = self.criterion(logits, targets)
                loss = raw_loss / self.accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            # Gradient Accumulation Boundary
            is_accumulation_boundary = (batch_idx + 1) % self.accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(loader)
            
            if is_accumulation_boundary or is_last_batch:
                self.scaler.unscale_(self.optimizer)
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.ema is not None:
                    self.ema.update(self.model)
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

    def evaluate_clips(
        self,
        models,
        records,
        cached_waveforms,
        frame_length=8000,
        frame_hop=None,
        frames_per_clip=15,
        drop_silent_tail_frames=False,
        sample_rate=16000,
        return_predictions=False,
    ):
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
        predictions = []
        
        if frame_hop is None:
            frame_hop = frame_length // 2

        # Pre-generate frame offsets
        offsets = [i * frame_hop for i in range(frames_per_clip)]
        
        with torch.no_grad():
            for path, frames in clips.items():
                label = frames[0]["label"]
                waveform_np = cached_waveforms[path]
                waveform = torch.from_numpy(waveform_np)
                duration_samples = None
                if drop_silent_tail_frames and "duration" in frames[0]:
                    duration_samples = int(float(frames[0]["duration"]) * sample_rate)
                    
                # Extract clip frames. Duration-aware eval avoids summing padding-only
                # frames for short events such as car horns.
                batch_frames = []
                for offset in offsets:
                    if duration_samples is not None and offset >= duration_samples:
                        continue
                    frame = waveform[:, offset:offset + frame_length]
                    if frame.shape[-1] < frame_length:
                        frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode='constant')
                    batch_frames.append(frame)
                if not batch_frames:
                    frame = waveform[:, :frame_length]
                    if frame.shape[-1] < frame_length:
                        frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode='constant')
                    batch_frames.append(frame)
                    
                # Shape: (valid_frames, 1, frame_length)
                batch_tensor = torch.stack(batch_frames).to(self.device)
                batch_tensor = self.transform_inputs(batch_tensor)
                
                # Forward pass through all ensembled models
                sum_probs = None
                for m in models:
                    with torch.amp.autocast(
                        device_type="cuda" if "cuda" in self.device.type else "cpu",
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    ):
                        logits = m(batch_tensor)
                        probs = F.softmax(logits, dim=-1)
                        if sum_probs is None:
                            sum_probs = torch.zeros_like(probs)
                        sum_probs += probs
                
                sum_probs /= len(models)
                
                # SUM rule: aggregate frame predictions
                clip_prob = torch.sum(sum_probs, dim=0) # (10,)
                predicted_class = torch.argmax(clip_prob).item()
                
                if return_predictions:
                    predictions.append({
                        "path": path,
                        "label": label,
                        "predicted": predicted_class
                    })
                
                if predicted_class == label:
                    correct += 1
                    
        if return_predictions:
            return correct / total, predictions
        else:
            return correct / total
