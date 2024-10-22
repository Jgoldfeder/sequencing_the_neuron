import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

class CD(Distiller):
    def __init__(self, students, teacher):
        super(CD, self).__init__()
        self.students = nn.ModuleList(students)
        self.teacher = teacher
        self.inputs = []
        self.outputs = []
        self.best = None
        self.pop_size = len(self.students)

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, image, target, **kwargs):
        # training function for the distillation method
        logits = self.forward_test(image)
        losses = [F.cross_entropy(logit, target) for logit in logits]
        return logits, losses

    def forward_test(self, image):
        logits = []
        for s in self.students:
            logits.append(s(image))
        return logits

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])
    
    def add_data(self,inputs,outputs,window = None):
        self.inputs.append(inputs)
        self.outputs.append(outputs)

    def get_adv(self, inputs, outputs):
        pass