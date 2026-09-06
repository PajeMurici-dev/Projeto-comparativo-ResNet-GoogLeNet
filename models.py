"""
Construção dos modelos ResNet-50 e GoogLeNet (Inception v1) via transfer learning
com torchvision. A última camada de classificação é substituída para o número
de classes do seu dataset (por padrão, 4: glioma, meningioma, pituitary, notumor).
"""

import torch.nn as nn
from torchvision import models

import config


def _freeze_all_but_head(model, head_params):
    """Congela todos os parâmetros, exceto os da camada final (head_params)."""
    for param in model.parameters():
        param.requires_grad = False
    for param in head_params:
        param.requires_grad = True


def build_resnet(num_classes: int, freeze_backbone: bool = config.FREEZE_BACKBONE):
    """
    ResNet-50 pré-treinada na ImageNet. A arquitetura usa blocos residuais
    (skip connections): a saída de um bloco é a soma de F(x) + x, em vez de
    só F(x). Isso permite treinar redes muito mais profundas (50 camadas
    aqui) sem sofrer tanto com vanishing gradient, porque o gradiente pode
    "pular" diretamente pelas conexões residuais durante a backpropagation.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        _freeze_all_but_head(model, model.fc.parameters())

    return model


def build_googlenet(num_classes: int, freeze_backbone: bool = config.FREEZE_BACKBONE):
    """
    GoogLeNet / Inception v1 pré-treinada na ImageNet.
    Importante: o GoogLeNet original tem duas saídas auxiliares (auxiliary
    classifiers) usadas só durante o treino, para combater vanishing gradient
    numa rede profunda de 22 camadas. Elas também precisam ter a camada final
    substituída, ou devem ser desativadas (aux_logits=False) se você não
    quiser lidar com as duas perdas extras.
    """
    weights = models.GoogLeNet_Weights.IMAGENET1K_V1
    # aux_logits=True mantém as saídas auxiliares originais do paper (Szegedy et al., 2015)
    model = models.googlenet(weights=weights, aux_logits=True, init_weights=False)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    head_params = list(model.fc.parameters())

    if model.aux1 is not None:
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
        head_params += list(model.aux1.fc2.parameters())
    if model.aux2 is not None:
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
        head_params += list(model.aux2.fc2.parameters())

    if freeze_backbone:
        _freeze_all_but_head(model, head_params)

    return model


def count_parameters(model):
    """Retorna (total_params, trainable_params) — útil para comparar custo computacional."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable