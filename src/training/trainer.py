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
        mixup_cfg=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.mixup_cfg = mixup_cfg or {}

    def train_epoch(self, loader):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
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
        return_predictions=False,
        aggregation="sum",
        sample_rate=16000,
        clip_seconds=4.0,
    ):
        """
        Evaluate whole audio clips by sliding a window over each waveform and
        aggregating per-frame softmax predictions.

        Aggregation rules (Abdoli et al. 2019, Sec. 2.4):
          * "sum"      — sum rule (Eq. 6 / Table 2 best for 16k)
          * "majority" — majority vote over argmax frame predictions (Eq. 5)
        """
        if frame_hop is None:
            frame_hop = frame_length // 2
        aggregation = (aggregation or "sum").lower()
        if aggregation not in {"sum", "majority"}:
            raise ValueError(f"Unknown aggregation '{aggregation}'. Use 'sum' or 'majority'.")

        for m in models:
            m.eval()

        # Group records by audio path (one entry per clip is enough for label)
        clips = {}
        for r in records:
            if r["path"] not in clips:
                clips[r["path"]] = r["label"]

        correct = 0
        total = len(clips)
        predictions = []

        # Sliding-window offsets over a fixed-length padded clip
        total_samples = int(sample_rate * clip_seconds)
        if total_samples <= frame_length:
            offsets = [0]
        else:
            max_start = total_samples - frame_length
            offsets = list(range(0, max_start + 1, frame_hop))

        num_frames = len(offsets)
        num_classes = 10

        with torch.no_grad():
            for path, label in clips.items():
                waveform = torch.from_numpy(cached_waveforms[path])

                batch_frames = []
                for offset in offsets:
                    frame = waveform[:, offset:offset + frame_length]
                    if frame.shape[-1] < frame_length:
                        frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode="constant")
                    batch_frames.append(frame)

                batch_tensor = torch.stack(batch_frames).to(self.device)

                # Average ensemble members first, then aggregate over frames
                sum_probs = torch.zeros((num_frames, num_classes), device=self.device)
                for m in models:
                    with torch.amp.autocast(
                        device_type="cuda" if "cuda" in self.device.type else "cpu",
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    ):
                        logits = m(batch_tensor)
                        sum_probs += F.softmax(logits, dim=-1)

                mean_probs = sum_probs / len(models)

                if aggregation == "sum":
                    # Eq. 6: y_i = (1/S) Σ o_ji  (mean is equivalent to sum for argmax)
                    clip_score = mean_probs.sum(dim=0)
                    predicted_class = int(torch.argmax(clip_score).item())
                else:
                    # Eq. 5: majority vote over frame-level argmax
                    frame_preds = torch.argmax(mean_probs, dim=-1)
                    votes = torch.bincount(frame_preds, minlength=num_classes)
                    predicted_class = int(torch.argmax(votes).item())

                if return_predictions:
                    predictions.append({
                        "path": path,
                        "label": label,
                        "predicted": predicted_class,
                    })

                if predicted_class == label:
                    correct += 1

        if return_predictions:
            return correct / total, predictions
        return correct / total
