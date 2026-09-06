"""
Script principal: treina e avalia ResNet-50 e GoogLeNet no mesmo dataset
(140k Real and Fake Faces - detecção de deepfakes), depois gera uma
tabela comparativa e gráficos para usar no artigo.

Uso:
    python compare_models.py
"""

import json
import os

import matplotlib.pyplot as plt

import config
from dataset import get_dataloaders
from models import build_resnet, build_googlenet
from train import run_experiment


def plot_training_curves(results_list, out_path):
    plt.figure(figsize=(8, 5))
    for res in results_list:
        epochs = range(1, len(res["history"]["val_acc"]) + 1)
        plt.plot(epochs, res["history"]["val_acc"], marker="o", label=res["model_name"])
    plt.xlabel("Época")
    plt.ylabel("Acurácia de validação")
    plt.title("Curva de acurácia de validação por época")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Gráfico salvo em {out_path}")


def print_comparison_table(results_list):
    header = f"{'Modelo':<12} {'Acc':<8} {'F1':<8} {'Tempo treino (s)':<18} {'Inferência (ms/img)':<20} {'Parâmetros':<15}"
    lines = [header, "-" * len(header)]
    for res in results_list:
        lines.append(
            f"{res['model_name']:<12} "
            f"{res['test_accuracy']:<8.4f} "
            f"{res['test_f1_macro']:<8.4f} "
            f"{res['total_train_time_seconds']:<18.1f} "
            f"{res['avg_inference_time_ms_per_image']:<20.2f} "
            f"{res['total_params']:<15,}"
        )
    table_str = "\n".join(lines)
    print("\n" + table_str)
    return table_str


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()
    num_classes = len(class_names)

    all_results = []

    # --- ResNet-50 ---
    resnet = build_resnet(num_classes)
    resnet_results = run_experiment("ResNet50", resnet, train_loader, val_loader, test_loader,
                                     class_names, class_weights)
    all_results.append(resnet_results)
    del resnet

    # --- GoogLeNet (Inception v1) ---
    googlenet = build_googlenet(num_classes)
    googlenet_results = run_experiment("GoogLeNet", googlenet, train_loader, val_loader, test_loader,
                                        class_names, class_weights)
    all_results.append(googlenet_results)
    del googlenet

    # --- Comparação final ---
    table_str = print_comparison_table(all_results)
    with open(os.path.join(config.RESULTS_DIR, "comparison_table.txt"), "w", encoding="utf-8") as f:
        f.write(table_str)

    plot_training_curves(all_results, os.path.join(config.RESULTS_DIR, "val_accuracy_comparison.png"))

    with open(os.path.join(config.RESULTS_DIR, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nExperimento concluído. Todos os resultados estão em: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()