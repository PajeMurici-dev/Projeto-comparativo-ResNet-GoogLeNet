"""
Script principal: treina e avalia ResNet-50 e GoogLeNet no mesmo dataset
(140k Real and Fake Faces - detecção de deepfakes), depois gera uma
tabela comparativa e gráficos para usar no artigo.

Uso:
    python compare_models.py
"""

import json
import os

import config
from dataset import get_dataloaders
from models import build_resnet, build_googlenet
from train import run_experiment
from reporting import plot_training_curves, plot_epoch_times, plot_loss_curves, plot_final_metrics_bar, print_comparison_table


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
    plot_epoch_times(all_results, os.path.join(config.RESULTS_DIR, "epoch_time_comparison.png"))
    plot_loss_curves(all_results, os.path.join(config.RESULTS_DIR, "loss_curves_comparison.png"))
    plot_final_metrics_bar(all_results, os.path.join(config.RESULTS_DIR, "final_metrics_bar.png"))

    with open(os.path.join(config.RESULTS_DIR, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nExperimento concluído. Todos os resultados estão em: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()