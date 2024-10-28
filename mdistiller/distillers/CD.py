import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller, Vanilla
from tqdm import tqdm

import sys
sys.path.append('../../reconstruction')
sys.path.append('../engine/utils.py')
from reconstruction.util import init_uniform

from mdistiller.engine.utils import log_msg
from .KD import kd_loss

class CD(nn.Module):
    def __init__(self, students, teacher, cfg):
        super(CD, self).__init__()
        # Students is a list of Vanilla distillers
        self.students = nn.ModuleList(students)
        self.teacher = teacher
        self.inputs = []
        self.outputs = []
        self.best = None
        self.cfg = cfg
        self.pop_size = len(self.students)

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for student in self.students:
            student.train(mode)
        self.teacher.eval()
        return self

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        params = []
        for student in self.students:
            params += student.get_learnable_parameters()
        return params

    def forward_train(self, image, target, **kwargs):
        # training function for the distillation method
        logits = []
        losses = {"kl": []}
        for student in self.students:
            student_logits, student_loss_dict = student.forward_train(image, target)
            logits.append(student_logits)
            losses["kl"].append(student_loss_dict["kl"])
        return logits, losses

    def forward_test(self, image):
        logits = []
        for s in self.students:
            logits.append(s.forward_test(image))
        return logits
    
    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])
    
    def add_data(self,inputs,outputs,window = None):
        self.inputs.append(inputs)
        self.outputs.append(outputs)

    def get_adv_samples(self, num_samples, base_images=None):
        # generate adversarial samples
        if self.cfg.DATASET.TYPE == "cifar100":
            input_dims = [num_samples, 3, 32, 32]
        else: 
            input_dims = [num_samples, 3, 224, 224]
        if self.cfg.CD.SAMPLE_INIT == "uniform":   
            adv = torch.zeros(input_dims, requires_grad = True, device="cuda")
            nn.init.uniform_(adv,-1,1)
        elif self.cfg.CD.SAMPLE_INIT == "image":
            adv = base_images.cuda()
            adv.requires_grad_()

        lr = self.cfg.CD.LR
        optimizer = torch.optim.Adam([adv], lr=lr)
        error=0
        softmax = torch.nn.Softmax()
        # pbar = tqdm(range(self.cfg.CD.EPOCHS))
        for epoch in range(self.cfg.CD.EPOCHS):
            if epoch in self.cfg.CD.SCHEDULE:
                lr = lr/10
                optimizer = torch.optim.Adam([adv], lr=lr)
            outs = []
            for idx,s in enumerate(self.students):
                s.cuda()
                out = torch.nn.functional.normalize(s.forward_test(image=adv), p=1.0, dim=-1)
                outs.append(out)
                s.zero_grad()
            outs = torch.stack(outs)
            outs = torch.transpose(outs,1,0).contiguous()
            dists = torch.cdist(outs,outs)
            # if error == 0:
            #     print("init. error:", -(dists.flatten().mean()))
                    
            error = -(dists.flatten().mean())
            #print(error)
            error.backward()
            optimizer.step()
            optimizer.zero_grad()
        #     pbar.set_description(log_msg(f"Epoch: {epoch}", "SAMPLING"))
        #     pbar.update()
        # pbar.close()

        # print("final error:",error)
        # print("stats:", adv.detach().abs().cpu().mean(),adv.detach().cpu().mean())
        return adv.detach().cpu(), error


class CDStudent(Vanilla):
    def __init__(self, student, cfg):
        super(CDStudent, self).__init__(student)
        self.temperature = cfg.KD.TEMPERATURE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)

        # losses
        loss = kd_loss(
            logits_student, target, self.temperature
        )
        losses_dict = {
            "kl": loss,
        }
        return logits_student, losses_dict