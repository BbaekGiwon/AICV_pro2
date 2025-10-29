import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # outputs: (batch_size, 4, 1024, 1024)
        # targets: (batch_size, 4, 1024, 1024)
        
        # Flatten outputs and targets to compute Dice coefficient
        outputs_flat = outputs.contiguous().view(outputs.size(0), outputs.size(1), -1)
        targets_flat = targets.contiguous().view(targets.size(0), targets.size(1), -1)

        # Compute intersection and union
        intersection = (outputs_flat * targets_flat).sum(dim=2)
        union = outputs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        # Compute Dice coefficient
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # Average over batch and classes
        dice_loss = 1 - dice_score.mean()

        return dice_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 가중치 파라미터: false positives에 대한 가중치
        self.beta = beta    # 가중치 파라미터: false negatives에 대한 가중치
        self.smooth = smooth

    def forward(self, outputs, targets):
        # outputs: (batch_size, 4, 1024, 1024)
        # targets: (batch_size, 4, 1024, 1024)
        # Flatten outputs and targets to compute Tversky coefficient
        outputs_flat = outputs.contiguous().view(outputs.size(0), outputs.size(1), -1)
        targets_flat = targets.contiguous().view(targets.size(0), targets.size(1), -1)

        # Compute true positives, false positives, and false negatives
        true_pos = (outputs_flat * targets_flat).sum(dim=2)
        false_pos = ((1 - targets_flat) * outputs_flat).sum(dim=2)
        false_neg = (targets_flat * (1 - outputs_flat)).sum(dim=2)

        # Compute Tversky index
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)

        # Tversky loss is 1 - Tversky index
        tversky_loss = 1 - tversky_index.mean()

        return tversky_loss
