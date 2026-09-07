"""
Treina APENAS a ResNet-50 e salva os resultados em results/ResNet50_results.json.

Use este script quando quiser treinar cada modelo em uma sessão separada
do Colab (por causa do limite de tempo de sessão) em vez de rodar os dois
de uma vez com compare_models.py.

Uso:
    python train_resnet.py
"""

import config
from dataset import get_dataloaders
from models import build_resnet
from train import run_experiment


def main():
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()
    num_classes = len(class_names)

    model = build_resnet(num_classes)
    run_experiment("ResNet50", model, train_loader, val_loader, test_loader,
                    class_names, class_weights)

    print("\n✅ Treino da ResNet-50 concluído. Resultados salvos em "
          f"{config.RESULTS_DIR}/ResNet50_results.json")
    print("Quando também tiver o GoogLeNet_results.json (rode train_googlenet.py "
          "em outra sessão), use generate_comparison.py para gerar a tabela final.")


if __name__ == "__main__":
    main()