import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class InceptionModule(nn.Module):
    """Single Inception block: four parallel branches concatenated channel-wise.

    Branch layout (as in the original GoogLeNet paper):
      - 1×1 convolution
      - 1×1 → 3×3 convolution
      - 1×1 → 5×5 convolution
      - 3×3 max-pool → 1×1 convolution
    """

    def __init__(self, in_channels: int, ch1x1: int, ch3x3_reduce: int,
                 ch3x3: int, ch5x5_reduce: int, ch5x5: int, pool_proj: int):
        super().__init__()

        # branch 1 — 1×1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # branch 2 — 1×1 reduce → 3×3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # branch 3 — 1×1 reduce → 5×5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # branch 4 — 3×3 max-pool → 1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class AuxiliaryClassifier(nn.Module):
    """Auxiliary classifier attached to intermediate layers during training.

    Used at Inception(4a) and Inception(4d). Helps combat vanishing gradients
    and provides additional regularization. Only used during training.
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.7):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    """GoogLeNet (Inception v1) implemented in pure PyTorch.

    Architecture follows the original paper "Going Deeper with Convolutions"
    (Szegedy et al., 2015), adapted for binary classification:

      Stem → 9 Inception modules → AvgPool → Dropout → Linear(1024→1) → Sigmoid

    Auxiliary classifiers are attached after Inception(4a) and Inception(4d)
    and contribute 0.3 weight each to the total training loss.

    Parameters
    ----------
    lr : float
        Learning rate for the Adam optimizer.
    weight_decay : float
        L2 regularization coefficient.
    use_dropout : bool
        Whether to apply dropout before the final classifier and inside
        auxiliary heads. Default True.
    dropout_prob : float
        Dropout probability. Default 0.4.
    aux_weight : float
        Weight multiplier for auxiliary classifier losses during training.
        Set to 0 to disable auxiliary heads entirely. Default 0.3.
    """

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4,
                 use_dropout: bool = True, dropout_prob: float = 0.4,
                 aux_weight: float = 0.3):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.aux_weight = aux_weight
        self.num_classes = 1  # binary → single logit + sigmoid

        # ---- stem ----
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ---- inception stack ----
        # inception 3a  (192 → 256)
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        # inception 3b  (256 → 480)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # inception 4a  (480 → 512)
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        # inception 4b  (512 → 512)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        # inception 4c  (512 → 512)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        # inception 4d  (512 → 528)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        # inception 4e  (528 → 832)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # inception 5a  (832 → 832)
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        # inception 5b  (832 → 1024)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # ---- auxiliary classifiers (only used during training) ----
        self.aux1 = AuxiliaryClassifier(512, self.num_classes, dropout_prob)
        self.aux2 = AuxiliaryClassifier(528, self.num_classes, dropout_prob)

        # ---- final classifier ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_prob) if use_dropout else nn.Identity()
        self.fc = nn.Linear(1024, self.num_classes)

        # ---- loss ----
        self.criterion = nn.BCEWithLogitsLoss()

        # ---- optimizer ----
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stem
        x = self.stem(x)

        # inception 3a, 3b → maxpool
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # inception 4a → aux1
        x = self.inception4a(x)
        aux1_out: Optional[torch.Tensor] = None
        if self.training and self.aux_weight > 0:
            aux1_out = self.aux1(x)

        # inception 4b, 4c, 4d → aux2
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2_out: Optional[torch.Tensor] = None
        if self.training and self.aux_weight > 0:
            aux2_out = self.aux2(x)

        # inception 4e → maxpool
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # inception 5a, 5b
        x = self.inception5a(x)
        x = self.inception5b(x)

        # final classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_weight > 0:
            return x, aux1_out, aux2_out
        return x

    def compute_loss(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        """Compute total loss including auxiliary classifiers.

        Parameters
        ----------
        outputs : Tensor or tuple
            Raw output from forward(). During training with aux heads this is
            (main_logits, aux1_logits, aux2_logits). Otherwise a single Tensor.
        targets : Tensor
            Ground-truth labels of shape (batch,) with float values in {0, 1}.
        """
        targets = targets.float().view(-1, 1)

        if isinstance(outputs, tuple):
            main_logits, aux1_logits, aux2_logits = outputs
            loss_main = self.criterion(main_logits, targets)
            loss_aux1 = self.criterion(aux1_logits, targets)
            loss_aux2 = self.criterion(aux2_logits, targets)
            return loss_main + self.aux_weight * (loss_aux1 + loss_aux2)
        else:
            return self.criterion(outputs, targets)

    def inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference and return probabilities + binary predictions.

        Parameters
        ----------
        x : Tensor
            Input batch of shape (N, 3, H, W).

        Returns
        -------
        probs : Tensor of shape (N, 1) — probability of class 1 (Tumor).
        preds : Tensor of shape (N,) — binary predictions (0 or 1).
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().squeeze(1)
        if was_training:
            self.train()
        return probs, preds

    def training_step(self, images: torch.Tensor,
                      labels: torch.Tensor) -> float:
        """Single training step: forward → loss → backward → optimizer step.

        Returns the loss value for logging.
        """
        self.train()
        self.optimizer.zero_grad()
        outputs = self.forward(images)
        loss = self.compute_loss(outputs, labels.float())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, images: torch.Tensor,
                        labels: torch.Tensor) -> Tuple[float, float]:
        """Single validation step: forward → loss → accuracy.

        Returns (loss, accuracy).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # discard aux for val
            loss = self.criterion(outputs, labels.float().view(-1, 1)).item()
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long().squeeze(1)
            acc = (preds == labels.long()).float().mean().item()
        return loss, acc
