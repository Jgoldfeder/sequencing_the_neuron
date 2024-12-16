import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .CDD import CDD
from .KD import KD, kd_loss


class CDD_iter(CDD):    
    def forward_train(self, image, augment=False):
        loss = 0
        if augment:
            lr = self.cfg.CD.LR
            if self.cfg.CD.RANDOM_INIT:
                augmented_image = nn.init.uniform_(torch.zeros_like(image, device="cuda", requires_grad=True), a=-1.0, b=1.0)
            else:
                augmented_image = image.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([augmented_image], lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            for epoch in range(self.cfg.CD.EPOCHS):
                logits_student, _ = self.student(augmented_image)
                logits_student = torch.nn.functional.normalize(logits_student, p=1.0, dim=-1)
                logits_teacher, _ = self.teacher(augmented_image)
                logits_teacher = torch.nn.functional.normalize(logits_teacher, p=1.0, dim=-1)
                self.student.zero_grad()
                self.teacher.zero_grad()
                loss = nn.MSELoss()(logits_student, logits_teacher)
                loss = -loss
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
        return logits_student, losses_dict, image, logits_teacher, loss