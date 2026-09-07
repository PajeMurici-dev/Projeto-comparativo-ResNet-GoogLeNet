"""
Junta os resultados já salvos de ResNet-50 e GoogLeNet (treinados em
sessões separadas com train_resnet.py e train_googlenet.py) e gera a
tabela comparativa final e o gráfico de curvas — sem precisar retreinar
nada.

Pré-requisito: os arquivos results/ResNet50_results.json e
results/GoogLeNet_results.json precisam já existir (persistidos, por
exemplo, no Google Drive, para sobreviverem entre sessões do Colab).

Uso:
    python generate_comparison.py
"""

import json
import os

import config
from reporting import plot_training_curves, print_comparison_table


def load_results(model_name: str):
    path = os.path.join(config.RESULTS_DIR, f"{model_name}_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Não encontrei {path}. Rode train_resnet.py e/ou train_googlenet.py "
            f"primeiro (em sessões separadas, se necessário)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    resnet_results = load_results("ResNet50")
    googlenet_results = load_results("GoogLeNet")

    all_results = [resnet_results, googlenet_results]

    table_str = print_comparison_table(all_results)
    with open(os.path.join(config.RESULTS_DIR, "comparison_table.txt"), "w", encoding="utf-8") as f:
        f.write(table_str)

    plot_training_curves(all_results, os.path.join(config.RESULTS_DIR, "val_accuracy_comparison.png"))

    with open(os.path.join(config.RESULTS_DIR, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nComparação final gerada em: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()