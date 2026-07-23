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
        hard_negative_margin_cfg=None,
        supervised_contrastive_cfg=None,
        distillation_cfg=None,
        teacher_model=None,
        ema=None,
        machinery_source_robust_cfg=None,
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
        self.hard_negative_margin_cfg = hard_negative_margin_cfg or {}
        self.hard_negative_pairs = self._build_hard_negative_pairs(self.hard_negative_margin_cfg)
        self.hard_negative_pair_specs = self._build_hard_negative_pair_specs(self.hard_negative_margin_cfg)
        self.supervised_contrastive_cfg = supervised_contrastive_cfg or {}
        self.distillation_cfg = distillation_cfg or {}
        self.machinery_source_robust_cfg = machinery_source_robust_cfg or {}
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad_(False)
        self.ema = ema

    @staticmethod
    def _build_hard_negative_pairs(cfg):
        if not cfg.get("enabled", False):
            return []

        pairs = []
        for group in cfg.get("groups", []):
            group = [int(class_id) for class_id in group]
            for target in group:
                for negative in group:
                    if negative != target:
                        pairs.append((target, negative))

        for pair in cfg.get("pairs", []):
            if len(pair) != 2:
                raise ValueError(f"hard_negative_margin pair must have 2 class ids, got {pair}")
            target, negative = int(pair[0]), int(pair[1])
            if target != negative:
                pairs.append((target, negative))

        unique_pairs = []
        for pair in pairs:
            if pair not in unique_pairs:
                unique_pairs.append(pair)
        return unique_pairs

    @staticmethod
    def _build_hard_negative_pair_specs(cfg):
        """List of (target, negative, relative_weight, margin) for optional per-pair scaling."""
        if not cfg.get("enabled", False):
            return []
        pairs = []
        for group in cfg.get("groups", []):
            group = [int(class_id) for class_id in group]
            for target in group:
                for negative in group:
                    if negative != target:
                        pairs.append((target, negative))
        for pair in cfg.get("pairs", []):
            if len(pair) != 2:
                raise ValueError(f"hard_negative_margin pair must have 2 class ids, got {pair}")
            target, negative = int(pair[0]), int(pair[1])
            if target != negative:
                pairs.append((target, negative))
        # dedupe preserving order
        unique = []
        for p in pairs:
            if p not in unique:
                unique.append(p)
        default_margin = float(cfg.get("margin", 0.5))
        pair_weights = cfg.get("pair_weights")
        pair_margins = cfg.get("pair_margins")
        specs = []
        for i, (t, n) in enumerate(unique):
            w = 1.0
            m = default_margin
            if isinstance(pair_weights, (list, tuple)) and i < len(pair_weights):
                w = float(pair_weights[i])
            if isinstance(pair_margins, (list, tuple)) and i < len(pair_margins):
                m = float(pair_margins[i])
            specs.append((t, n, w, m))
        return specs

    def hard_negative_margin_loss(self, logits, targets):
        cfg = self.hard_negative_margin_cfg
        specs = self.hard_negative_pair_specs
        if not cfg.get("enabled", False) or not specs:
            return logits.new_zeros(())

        losses = []
        weights = []
        for target_class, negative_class, rel_weight, margin in specs:
            mask = targets == target_class
            if mask.any():
                target_logits = logits[mask, target_class]
                negative_logits = logits[mask, negative_class]
                losses.append(F.relu(negative_logits - target_logits + margin).mean())
                weights.append(float(rel_weight))

        if not losses:
            return logits.new_zeros(())
        stacked = torch.stack(losses)
        w = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
        w = w / w.sum().clamp_min(1e-6)
        return (stacked * w).sum()

    def machinery_source_robust_loss(self, logits, targets, source_ids):
        """Source-group robust CE on machinery classes + optional use of pair term externally."""
        cfg = self.machinery_source_robust_cfg
        if not cfg.get("enabled", False) or source_ids is None:
            return logits.new_zeros(())

        machinery = {int(x) for x in cfg.get("machinery_classes", [0, 4, 5, 7])}
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for c in machinery:
            mask = mask | (targets == c)
        if not mask.any():
            return logits.new_zeros(())

        # per-sample CE on machinery subset
        per = F.cross_entropy(logits.float(), targets, reduction="none")
        m_logits_ids = source_ids[mask]
        m_losses = per[mask]
        # group by source id
        unique = torch.unique(m_logits_ids)
        group_means = []
        for s in unique.tolist():
            sm = m_logits_ids == s
            if sm.any():
                group_means.append(m_losses[sm].mean())
        if not group_means:
            return logits.new_zeros(())
        stacked = torch.stack(group_means)
        tau = max(float(cfg.get("source_temperature", 0.1)), 1e-4)
        # smooth max over train sources present in batch
        return tau * torch.logsumexp(stacked / tau, dim=0)


    def supervised_contrastive_loss(self, features, targets, source_ids=None):
        cfg = self.supervised_contrastive_cfg
        if not cfg.get("enabled", False):
            return features.new_zeros(())

        features = F.normalize(features.float(), dim=1)
        targets = targets.view(-1, 1)
        positive_mask = torch.eq(targets, targets.T)

        if bool(cfg.get("source_aware", True)) and source_ids is not None:
            source_ids = source_ids.view(-1, 1)
            positive_mask = positive_mask & torch.ne(source_ids, source_ids.T)

        logits_mask = torch.ones_like(positive_mask, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)
        positive_mask = positive_mask & logits_mask

        positive_counts = positive_mask.sum(dim=1)
        valid_anchors = positive_counts > 0
        if not valid_anchors.any():
            return features.new_zeros(())

        temperature = float(cfg.get("temperature", 0.1))
        similarity = torch.matmul(features, features.T) / max(temperature, 1e-6)
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        exp_similarity = torch.exp(similarity) * logits_mask.float()
        log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
        return -mean_log_prob_pos[valid_anchors].mean()

    def protected_distillation_loss(self, student_logits, teacher_logits, targets):
        cfg = self.distillation_cfg
        if not cfg.get("enabled", False) or self.teacher_model is None:
            return student_logits.new_zeros(())

        temperature = max(float(cfg.get("temperature", 2.0)), 1e-6)
        protect_classes = cfg.get("protect_classes", [])
        if protect_classes:
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for class_id in protect_classes:
                mask = mask | (targets == int(class_id))
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)

        if not mask.any():
            return student_logits.new_zeros(())

        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=1)
        per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
        per_sample = per_sample * temperature * temperature
        return per_sample[mask].mean()

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
        
        supcon_enabled = bool(self.supervised_contrastive_cfg.get("enabled", False))

        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                inputs, targets, source_ids = batch
            else:
                inputs, targets = batch
                source_ids = None

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if source_ids is not None:
                source_ids = source_ids.to(self.device, non_blocking=True)
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
                if supcon_enabled:
                    logits, features = self.model(mixed_inputs, return_features=True)
                else:
                    logits = self.model(mixed_inputs)
                    features = None
                if mixup_enabled and lam < 1.0:
                    raw_loss = lam * self.criterion(logits, targets_a) + (1.0 - lam) * self.criterion(logits, targets_b)
                else:
                    raw_loss = self.criterion(logits, targets)
                supcon_cfg = self.supervised_contrastive_cfg
                if supcon_cfg.get("enabled", False):
                    apply_to_mixup = bool(supcon_cfg.get("apply_to_mixup", False))
                    if lam >= 1.0 or apply_to_mixup:
                        supcon_loss = self.supervised_contrastive_loss(features, targets, source_ids)
                    else:
                        supcon_loss = logits.new_zeros(())
                    raw_loss = raw_loss + float(supcon_cfg.get("weight", 0.05)) * supcon_loss
                hard_negative_cfg = self.hard_negative_margin_cfg
                if hard_negative_cfg.get("enabled", False):
                    apply_to_mixup = bool(hard_negative_cfg.get("apply_to_mixup", False))
                    if mixup_enabled and lam < 1.0 and apply_to_mixup:
                        margin_loss = (
                            lam * self.hard_negative_margin_loss(logits, targets_a)
                            + (1.0 - lam) * self.hard_negative_margin_loss(logits, targets_b)
                        )
                    elif lam >= 1.0:
                        margin_loss = self.hard_negative_margin_loss(logits, targets)
                    else:
                        margin_loss = logits.new_zeros(())
                    raw_loss = raw_loss + float(hard_negative_cfg.get("weight", 0.05)) * margin_loss
                msr_cfg = self.machinery_source_robust_cfg
                if msr_cfg.get("enabled", False):
                    apply_to_mixup = bool(msr_cfg.get("apply_to_mixup", False))
                    if mixup_enabled and lam < 1.0 and not apply_to_mixup:
                        src_loss = logits.new_zeros(())
                    else:
                        # use non-mixed targets for source grouping
                        src_loss = self.machinery_source_robust_loss(logits, targets, source_ids)
                    raw_loss = raw_loss + float(msr_cfg.get("source_weight", 0.15)) * src_loss
                distillation_cfg = self.distillation_cfg
                if distillation_cfg.get("enabled", False) and self.teacher_model is not None:
                    apply_to_mixup = bool(distillation_cfg.get("apply_to_mixup", False))
                    if lam >= 1.0 or apply_to_mixup:
                        with torch.no_grad():
                            teacher_logits = self.teacher_model(mixed_inputs)
                        if mixup_enabled and lam < 1.0 and apply_to_mixup:
                            distill_loss = (
                                lam * self.protected_distillation_loss(logits, teacher_logits, targets_a)
                                + (1.0 - lam) * self.protected_distillation_loss(logits, teacher_logits, targets_b)
                            )
                        else:
                            distill_loss = self.protected_distillation_loss(logits, teacher_logits, targets)
                    else:
                        distill_loss = logits.new_zeros(())
                    raw_loss = raw_loss + float(distillation_cfg.get("weight", 0.2)) * distill_loss
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
            
        # Group records by logical clip. Most datasets use one path per clip;
        # Speech Commands silence windows can share one background-noise path.
        clips = defaultdict(list)
        for r in records:
            clips[r.get("clip_id", r["path"])].append(r)
            
        correct = 0
        total = len(clips)
        predictions = []
        
        if frame_hop is None:
            frame_hop = frame_length // 2

        # Pre-generate frame offsets
        offsets = [i * frame_hop for i in range(frames_per_clip)]
        
        with torch.no_grad():
            for _, frames in clips.items():
                label = frames[0]["label"]
                path = frames[0]["path"]
                waveform_np = cached_waveforms[path]
                waveform = torch.from_numpy(waveform_np)
                duration_samples = None
                if drop_silent_tail_frames and "duration" in frames[0]:
                    duration_samples = int(float(frames[0]["duration"]) * sample_rate)
                base_frame_start = int(frames[0].get("frame_start", 0))
                    
                # Extract clip frames. Duration-aware eval avoids summing padding-only
                # frames for short events such as car horns.
                batch_frames = []
                for offset in offsets:
                    if duration_samples is not None and offset >= duration_samples:
                        continue
                    absolute_offset = base_frame_start + offset
                    frame = waveform[:, absolute_offset:absolute_offset + frame_length]
                    if frame.shape[-1] < frame_length:
                        frame = F.pad(frame, (0, frame_length - frame.shape[-1]), mode='constant')
                    batch_frames.append(frame)
                if not batch_frames:
                    frame = waveform[:, base_frame_start:base_frame_start + frame_length]
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
                        "clip_id": frames[0].get("clip_id", path),
                        "label": label,
                        "predicted": predicted_class
                    })
                
                if predicted_class == label:
                    correct += 1
                    
        if return_predictions:
            return correct / total, predictions
        else:
            return correct / total
