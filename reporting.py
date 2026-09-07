"""
Funções de geração de relatório (tabela comparativa e gráfico de curvas),
compartilhadas entre compare_models.py (fluxo completo numa sessão só) e
generate_comparison.py (fluxo dividido em sessões separadas).
"""

import matplotlib.pyplot as plt


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