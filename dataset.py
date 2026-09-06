"""
Carregamento do dataset "140k Real and Fake Faces" usando ImageFolder.

Diferente do HAM10000, este dataset já vem organizado em pastas por
classe (real/ e fake/) e já vem dividido em train/valid/test — então
o carregamento aqui é mais direto, parecido com o do dataset original
de tumores cerebrais.
"""

import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _maybe_subset(dataset, fraction: float, seed: int):
    """
    Se fraction < 1.0, retorna um Subset com uma fração aleatória do
    dataset (mantendo a proporção de classes, já que a amostragem é
    uniforme sobre um dataset já balanceado). Usado para acelerar
    experimentos em ambientes com pouco poder computacional.
    """
    if fraction >= 1.0:
        return dataset
    n_total = len(dataset)
    n_subset = int(n_total * fraction)
    rng = random.Random(seed)
    indices = rng.sample(range(n_total), n_subset)
    return Subset(dataset, indices)


def compute_class_weights(dataset, num_classes: int):
    """
    Calcula pesos inversamente proporcionais à frequência de cada classe.
    Funciona tanto com ImageFolder quanto com Subset de um ImageFolder.
    """
    if isinstance(dataset, Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets

    counts = [0] * num_classes
    for t in targets:
        counts[t] += 1

    total = len(targets)
    weights = [total / (num_classes * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float)


def get_dataloaders():
    """
    Retorna train_loader, val_loader, test_loader, class_names e os pesos
    de classe (None se config.USE_CLASS_WEIGHTS=False).
    """
    train_set = datasets.ImageFolder(config.TRAIN_DIR, transform=build_transforms(train=True))
    val_set = datasets.ImageFolder(config.VAL_DIR, transform=build_transforms(train=False))
    test_set = datasets.ImageFolder(config.TEST_DIR, transform=build_transforms(train=False))

    class_names = train_set.classes  # ex: ['fake', 'real'], ordem alfabética

    train_set = _maybe_subset(train_set, config.SUBSET_FRACTION, config.SEED)
    val_set = _maybe_subset(val_set, config.SUBSET_FRACTION, config.SEED + 1)
    test_set = _maybe_subset(test_set, config.SUBSET_FRACTION, config.SEED + 2)

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_set, len(class_names))

    print(f"Classes: {class_names}")
    print(f"Treino: {len(train_set)} | Validação: {len(val_set)} | Teste: {len(test_set)}")
    if config.SUBSET_FRACTION < 1.0:
        print(f"(usando {config.SUBSET_FRACTION * 100:.0f}% do dataset original em cada split)")

    return train_loader, val_loader, test_loader, class_names, class_weights