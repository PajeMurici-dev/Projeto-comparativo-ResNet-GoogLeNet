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


def plot_epoch_times(results_list, out_path):
    """Tempo de treino de cada época, por modelo — mostra o custo computacional ao longo do treino."""
    plt.figure(figsize=(8, 5))
    plotted_any = False
    for res in results_list:
        epoch_times = res["history"].get("epoch_time_seconds")
        if not epoch_times:
            print(f"⚠️  '{res['model_name']}' não tem epoch_time_seconds salvo "
                  f"(resultado gerado antes dessa métrica existir) — pulando no gráfico. "
                  f"Retreine com o train.py atualizado para incluir essa métrica.")
            continue
        epochs = range(1, len(epoch_times) + 1)
        plt.plot(epochs, epoch_times, marker="o", label=res["model_name"])
        plotted_any = True

    if not plotted_any:
        plt.close()
        print("Nenhum resultado tinha tempo por época salvo — gráfico não gerado.")
        return

    plt.xlabel("Época")
    plt.ylabel("Tempo de treino (segundos)")
    plt.title("Tempo de treino por época")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Gráfico salvo em {out_path}")


def plot_loss_curves(results_list, out_path):
    """Curvas de loss de treino e validação — ajuda a identificar overfitting/underfitting."""
    plt.figure(figsize=(8, 5))
    for res in results_list:
        epochs = range(1, len(res["history"]["train_loss"]) + 1)
        plt.plot(epochs, res["history"]["train_loss"], linestyle="--", marker="o",
                  label=f"{res['model_name']} (treino)")
        plt.plot(epochs, res["history"]["val_loss"], linestyle="-", marker="s",
                  label=f"{res['model_name']} (validação)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curvas de loss (treino vs. validação)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Gráfico salvo em {out_path}")


def plot_final_metrics_bar(results_list, out_path):
    """
    Gráfico de barras comparando acurácia e F1 finais de teste entre os modelos,
    lado a lado — bom resumo visual para a seção de Resultados.
    """
    model_names = [res["model_name"] for res in results_list]
    accuracies = [res["test_accuracy"] for res in results_list]
    f1_scores = [res["test_f1_macro"] for res in results_list]

    x = range(len(model_names))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar([i - width / 2 for i in x], accuracies, width, label="Acurácia")
    plt.bar([i + width / 2 for i in x], f1_scores, width, label="F1 (macro)")
    plt.xticks(list(x), model_names)
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Acurácia e F1 finais no conjunto de teste")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
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