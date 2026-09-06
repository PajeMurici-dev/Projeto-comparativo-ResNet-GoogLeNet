"""
Loop de treino/validação/teste, genérico para ResNet-50 e GoogLeNet.
Calcula acurácia, F1-score (macro), matriz de confusão, tempo de treino
e tempo de inferência — as métricas usadas na seção de Resultados do artigo.
"""

import time
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import config


def train_one_epoch(model, loader, optimizer, criterion, is_googlenet: bool):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        optimizer.zero_grad()

        if is_googlenet:
            # Em modo de treino com aux_logits=True, o GoogLeNet retorna
            # (saída principal, saída_aux1, saída_aux2). O paper original
            # soma as 3 perdas, ponderando as auxiliares por 0.3 cada.
            main_out, aux1_out, aux2_out = model(images)
            loss = criterion(main_out, labels) \
                + 0.3 * criterion(aux1_out, labels) \
                + 0.3 * criterion(aux2_out, labels)
            outputs = main_out
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Usado para validação e teste (GoogLeNet em eval() só retorna a saída principal)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / total
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1, all_preds, all_labels


@torch.no_grad()
def measure_inference_time(model, loader, n_batches: int = 20):
    """Tempo médio de inferência por imagem (ms) — métrica de eficiência computacional."""
    model.eval()
    times = []
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        images = images.to(config.DEVICE)
        start = time.perf_counter()
        _ = model(images)
        if config.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed / images.size(0))
    return sum(times) / len(times) * 1000  # ms por imagem


def run_experiment(model_name: str, model, train_loader, val_loader, test_loader, class_names,
                    class_weights=None):
    """
    Executa o pipeline completo para um modelo: treino, validação por época,
    avaliação final no teste, e salva um JSON com todas as métricas.

    class_weights: tensor opcional com um peso por classe, usado para
    compensar datasets desbalanceados (ex: HAM10000, onde 'nv' domina).
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    model = model.to(config.DEVICE)
    is_googlenet = model_name.lower().startswith("googlenet")

    if class_weights is not None and config.USE_CLASS_WEIGHTS:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print(f"\n==== Treinando {model_name} ====")
    start_train = time.perf_counter()

    best_val_f1 = 0.0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, is_googlenet)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"[{model_name}] Época {epoch}/{config.NUM_EPOCHS} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt"))

    total_train_time = time.perf_counter() - start_train

    # Avaliação final no conjunto de teste
    test_loss, test_acc, test_f1, preds, labels = evaluate(model, test_loader, criterion)
    cm = confusion_matrix(labels, preds).tolist()
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)

    inference_ms = measure_inference_time(model, test_loader)

    from models import count_parameters
    total_params, trainable_params_count = count_parameters(model)

    results = {
        "model_name": model_name,
        "num_classes": len(class_names),
        "class_names": class_names,
        "epochs": config.NUM_EPOCHS,
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "total_train_time_seconds": total_train_time,
        "avg_inference_time_ms_per_image": inference_ms,
        "total_params": total_params,
        "trainable_params": trainable_params_count,
    }

    out_path = os.path.join(config.RESULTS_DIR, f"{model_name}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[{model_name}] Resultados salvos em {out_path}")
    print(f"[{model_name}] Teste -> acc={test_acc:.4f} f1={test_f1:.4f} "
          f"| tempo_treino={total_train_time:.1f}s | inferência={inference_ms:.2f}ms/img "
          f"| params={total_params:,} (treináveis: {trainable_params_count:,})")

    return results