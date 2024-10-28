import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .KD import KD, kd_loss


class CDD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(CDD, self).__init__(student, teacher)
        self.cfg = cfg
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
    
    def forward_train(self, image, target, **kwargs):
        lr = self.cfg.CD.LR
        augmented_image = image.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([augmented_image], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        for epoch in range(self.cfg.CD.EPOCHS):
            logits_student, _ = self.student(augmented_image)
            logits_student = torch.nn.functional.normalize(logits_student, p=1.0, dim=-1)
            logits_teacher, _ = self.teacher(image)
            logits_teacher = torch.nn.functional.normalize(logits_teacher, p=1.0, dim=-1)
            self.student.zero_grad()
            self.teacher.zero_grad()
            loss = nn.MSELoss()(logits_student, logits_teacher)
            augmented_image.grad = torch.autograd.grad(loss, augmented_image)[0]
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

        image = augmented_image.detach().requires_grad_(False)


        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )
        losses_dict = {
            # "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
