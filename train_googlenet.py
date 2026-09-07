"""
Treina APENAS o GoogLeNet e salva os resultados em results/GoogLeNet_results.json.

Use este script quando quiser treinar cada modelo em uma sessão separada
do Colab (por causa do limite de tempo de sessão) em vez de rodar os dois
de uma vez com compare_models.py.

Uso:
    python train_googlenet.py
"""

import config
from dataset import get_dataloaders
from models import build_googlenet
from train import run_experiment


def main():
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()
    num_classes = len(class_names)

    model = build_googlenet(num_classes)
    run_experiment("GoogLeNet", model, train_loader, val_loader, test_loader,
                    class_names, class_weights)

    print("\n✅ Treino do GoogLeNet concluído. Resultados salvos em "
          f"{config.RESULTS_DIR}/GoogLeNet_results.json")
    print("Quando também tiver o ResNet50_results.json (rode train_resnet.py "
          "em outra sessão), use generate_comparison.py para gerar a tabela final.")


if __name__ == "__main__":
    main()