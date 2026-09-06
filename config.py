"""
Configurações centrais do experimento: caminhos, hiperparâmetros e device.
Ajustado para o dataset "140k Real and Fake Faces" (detecção de deepfakes).

Esse dataset já vem organizado em pastas por classe (formato ImageFolder),
e já vem dividido em train/valid/test — não precisamos fazer split manual
como fizemos com o HAM10000.
"""

import os
import torch

# --- Caminhos do dataset ---
# Estrutura esperada após rodar download_data.py:
#   ./data/train/real/  ./data/train/fake/
#   ./data/valid/real/  ./data/valid/fake/
#   ./data/test/real/   ./data/test/fake/
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")

# --- Hiperparâmetros ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
SEED = 42

# --- Desbalanceamento de classes ---
# Este dataset já vem perfeitamente balanceado (70k reais / 70k falsas),
# então pesos de classe não são necessários por padrão. Deixamos a opção
# disponível caso você use um subconjunto que acabe desbalanceado.
USE_CLASS_WEIGHTS = False

# --- Subamostragem opcional ---
# O dataset completo tem 140k imagens (~4GB) e pode ser pesado para treinar
# em ambientes gratuitos (ex: Colab). Se SUBSET_FRACTION < 1.0, cada split
# (treino/validação/teste) é reduzido aleatoriamente a essa fração,
# mantendo a proporção original entre as classes.
SUBSET_FRACTION = 1.0  # ex: 0.2 usa só 20% de cada split

# --- Transfer learning ---
FREEZE_BACKBONE = True

# --- Saídas ---
RESULTS_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")