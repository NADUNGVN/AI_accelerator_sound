import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeAttentionModule(nn.Module):
    """
    TAM (Time Attention Module):
    Projects channels to 1 to compute temporal importance weights,
    applies them to a locally convolved version of the input, and adds a residual.
    """
    def __init__(self, channels):
        super().__init__()
        self.f_triple_prime = nn.Conv1d(channels, 1, kernel_size=1)
        self.f_s = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: (batch, channels, length)
        s = self.f_triple_prime(x)        # (batch, 1, length)
        s_prime = torch.sigmoid(s)        # (batch, 1, length)
        
        n = self.f_s(x) * s_prime         # (batch, channels, length)
        return x + n

class ChannelAttentionModule(nn.Module):
    """
    CAM (Channel Attention Module):
    Squeezes temporal info, passes through a bottleneck MLP (Squeeze-and-Excitation),
    and scales channels adaptively.
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduction = max(1, channels // reduction_ratio)
        self.fc1 = nn.Conv1d(channels, reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(reduction, channels, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, channels, length)
        z = torch.mean(x, dim=-1, keepdim=True)  # (batch, channels, 1)
        
        z_prime = torch.relu(self.fc1(z))
        z_prime = torch.sigmoid(self.fc2(z_prime))  # (batch, channels, 1)
        
        m = x * z_prime
        return x + m

class TCAMBlock(nn.Module):
    """
    Combines TAM and CAM sequentially.
    """
    def __init__(self, channels):
        super().__init__()
        self.tam = TimeAttentionModule(channels)
        self.cam = ChannelAttentionModule(channels)

    def forward(self, x):
        return self.cam(self.tam(x))

class TCAM1DCNN(nn.Module):
    """
    TCAM1DCNN Architecture from the 2024 ESWA paper.
    6 stages of Conv1D + TAM + CAM, followed by a final Conv1D, GAP, and Softmax.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Stage 1: input 8000 x 1 -> output 8000 x 32
        self.conv1 = nn.Conv1d(1, 32, kernel_size=32, stride=1, padding=15)
        self.tcam1 = TCAMBlock(32)
        
        # Stage 2: input 8000 x 32 -> output 4000 x 32
        self.conv2 = nn.Conv1d(32, 32, kernel_size=16, stride=2, padding=7)
        self.tcam2 = TCAMBlock(32)
        
        # Stage 3: input 4000 x 32 -> output 2000 x 64
        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4)
        self.tcam3 = TCAMBlock(64)
        
        # Stage 4: input 2000 x 64 -> output 1000 x 64
        self.conv4 = nn.Conv1d(64, 64, kernel_size=6, stride=2, padding=2)
        self.tcam4 = TCAMBlock(64)
        
        # Stage 5: input 1000 x 64 -> output 200 x 128
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, stride=5, padding=1)
        self.tcam5 = TCAMBlock(128)
        
        # Stage 6: input 200 x 128 -> output 40 x 128
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, stride=5, padding=1)
        self.tcam6 = TCAMBlock(128)
        
        # Final classification stage: input 40 x 128 -> output 20 x 256
        self.conv7 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Final classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input shape: (batch, 1, 8000)
        x = F.relu(self.conv1(x))
        x = self.tcam1(x)
        
        x = F.relu(self.conv2(x))
        x = self.tcam2(x)
        
        x = F.relu(self.conv3(x))
        x = self.tcam3(x)
        
        x = F.relu(self.conv4(x))
        x = self.tcam4(x)
        
        x = F.relu(self.conv5(x))
        x = self.tcam5(x)
        
        x = F.relu(self.conv6(x))
        x = self.tcam6(x)
        
        x = F.relu(self.conv7(x))
        
        # Global Average Pooling
        x = torch.mean(x, dim=-1)
        
        # Output Logits
        logits = self.fc(x)
        return logits
