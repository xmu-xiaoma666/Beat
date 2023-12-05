import imp
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class classifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(classifier, self).__init__()

        self.block = nn.Linear(input_dim, output_dim)
        self.block.apply(weights_init_classifier)

    def forward(self, x):
        x = self.block(x)
        return x


class Id_Loss(nn.Module):

    def __init__(self, opt, part, feature_length):
        super(Id_Loss, self).__init__()

        self.opt = opt
        self.part = part

        W = []
        for i in range(part):
            W.append(classifier(feature_length, opt.class_num))
        self.W = nn.Sequential(*W)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def calculate_IdLoss(self, image_embedding_local, text_embedding_local, label):

        label = label.view(label.size(0))
        
        Lipt_local = 0
        Ltpi_local = 0

        for i in range(self.part):

            score_i2t_local_i = self.W[i](image_embedding_local[:, :, i])
            score_t2i_local_i = self.W[i](text_embedding_local[:, :, i])
            

            Lipt_local += self.criterion(score_i2t_local_i, label)
            Ltpi_local += self.criterion(score_t2i_local_i, label)

        loss = (Lipt_local + Ltpi_local) / self.part

        return loss

    def forward(self, image_embedding_local, text_embedding_local, label):

        loss = self.calculate_IdLoss(image_embedding_local, text_embedding_local, label)

        return loss



class Id_Loss_max(nn.Module):

    def __init__(self, opt, part, feature_length):
        super(Id_Loss_max, self).__init__()

        self.opt = opt
        self.part = part

        W = []
        for i in range(part):
            W.append(classifier(feature_length, opt.class_num))
        self.W = nn.Sequential(*W)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def calculate_IdLoss(self, image_embedding_local, text_embedding_local, label):

        label = label.view(label.size(0))
        
        

        Lipt_local = 0
        Ltpi_local = 0

        for i in range(self.part):

            score_i2t_local_i = self.W[i](image_embedding_local[:, :, i])
            score_t2i_local_i = self.W[i](text_embedding_local[:, :, i])
            

            Lipt_local += self.criterion(score_i2t_local_i, label)
            Ltpi_local += self.criterion(score_t2i_local_i, label)
        
        loss = max(Lipt_local , Ltpi_local) / self.part

        return loss

    def forward(self, image_embedding_local, text_embedding_local, label):

        loss = self.calculate_IdLoss(image_embedding_local, text_embedding_local, label)

        return loss


class Id_Loss_min(nn.Module):

    def __init__(self, opt, part, feature_length):
        super(Id_Loss_min, self).__init__()

        self.opt = opt
        self.part = part

        W = []
        for i in range(part):
            W.append(classifier(feature_length, opt.class_num))
        self.W = nn.Sequential(*W)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def calculate_IdLoss(self, image_embedding_local, text_embedding_local, label):

        label = label.view(label.size(0))
        
        

        Lipt_local = 0
        Ltpi_local = 0

        for i in range(self.part):

            score_i2t_local_i = self.W[i](image_embedding_local[:, :, i])
            score_t2i_local_i = self.W[i](text_embedding_local[:, :, i])
            

            Lipt_local += self.criterion(score_i2t_local_i, label)
            Ltpi_local += self.criterion(score_t2i_local_i, label)

        loss = min(Lipt_local , Ltpi_local) / self.part

        return loss

    def forward(self, image_embedding_local, text_embedding_local, label):

        loss = self.calculate_IdLoss(image_embedding_local, text_embedding_local, label)

        return loss

class Id_Loss_KL(nn.Module):

    def __init__(self, opt, part, feature_length):
        super(Id_Loss_KL, self).__init__()

        self.opt = opt
        self.part = part

        W = []
        for i in range(part):
            W.append(classifier(feature_length, opt.class_num))
        self.W = nn.Sequential(*W)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss()

    def calculate_IdLoss(self, image_embedding_local, text_embedding_local, label):

        label = label.view(label.size(0))
        temp = 10
        weight = 1

        

        Lipt_local = 0
        Ltpi_local = 0
        Lipt_local_kl = 0
        Ltpi_local_kl = 0

        for i in range(self.part):

            score_i2t_local_i = self.W[i](image_embedding_local[:, :, i])
            score_t2i_local_i = self.W[i](text_embedding_local[:, :, i])

            Lipt_local += self.criterion(score_i2t_local_i, label)
            Ltpi_local += self.criterion(score_t2i_local_i, label)
            Lipt_local_kl += self.kl_loss(F.log_softmax(score_i2t_local_i / temp, dim=-1),F.softmax(score_t2i_local_i / temp, dim=-1))
            Ltpi_local_kl += self.kl_loss(F.log_softmax(score_t2i_local_i / temp, dim=-1),F.softmax(score_i2t_local_i / temp, dim=-1))
            

        loss = (Lipt_local + Ltpi_local + weight*(Lipt_local_kl + Ltpi_local_kl)) / self.part

        return loss

    def forward(self, image_embedding_local, text_embedding_local, label):

        loss = self.calculate_IdLoss(image_embedding_local, text_embedding_local, label)

        return loss

