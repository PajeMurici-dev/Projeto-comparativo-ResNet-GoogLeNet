"""
Baixa o dataset "140k Real and Fake Faces" via kagglehub e organiza os
arquivos na estrutura que dataset.py espera:
    ./data/train/{real,fake}/
    ./data/valid/{real,fake}/
    ./data/test/{real,fake}/

O dataset original no Kaggle costuma vir aninhado (algo como
real_vs_fake/real-vs-fake/train/...), então este script localiza as
pastas train/valid/test automaticamente, verificando que cada uma
contém subpastas 'real' e 'fake', e copia só o conteúdo relevante para
fora do aninhamento.

Uso:
    python download_data.py
"""

import os
import shutil

import kagglehub

DATASET_ID = "xhlulu/140k-real-and-fake-faces"
TARGET_DIR = "./data"

# Nomes possíveis para cada split (o dataset usa 'valid', mas outros
# datasets parecidos às vezes usam 'val')
SPLIT_NAME_CANDIDATES = {
    "train": ["train"],
    "valid": ["valid", "val"],
    "test": ["test"],
}


def print_tree(root: str, max_depth: int = 3):
    root = root.rstrip(os.sep)
    root_depth = root.count(os.sep)
    for current_dir, subdirs, files in os.walk(root):
        depth = current_dir.count(os.sep) - root_depth
        if depth > max_depth:
            subdirs[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(current_dir) or current_dir}/")
        if depth == max_depth:
            continue
        for f in files[:3]:
            print(f"{indent}  {f}")
        if len(files) > 3:
            print(f"{indent}  ... (+{len(files) - 3} arquivos)")


def find_split_dir(root: str, name_candidates):
    """
    Procura, em toda a árvore de diretórios, uma pasta cujo nome esteja em
    name_candidates E que contenha subpastas 'real' e 'fake' — evitando
    confundir com pastas de mesmo nome em outro contexto.
    """
    for dirpath, dirnames, _ in os.walk(root):
        base_name = os.path.basename(dirpath).lower()
        if base_name in name_candidates:
            children_lower = {d.lower() for d in dirnames}
            if "real" in children_lower and "fake" in children_lower:
                return dirpath
    return None


def main():
    print(f"Baixando dataset '{DATASET_ID}' via kagglehub (são ~4GB, pode demorar)...")
    download_path = kagglehub.dataset_download(DATASET_ID)
    print(f"\nDataset baixado em: {download_path}\n")

    print("Estrutura de diretórios encontrada (até 3 níveis):")
    print_tree(download_path)

    found = {}
    for split_name, candidates in SPLIT_NAME_CANDIDATES.items():
        src = find_split_dir(download_path, candidates)
        found[split_name] = src

    missing = [name for name, src in found.items() if src is None]
    if missing:
        print(f"\n⚠️  Não encontrei automaticamente a(s) pasta(s): {missing}. "
              f"Confira a árvore acima e ajuste TRAIN_DIR/VAL_DIR/TEST_DIR "
              f"em config.py manualmente, apontando para dentro de:")
        print(f"    {download_path}")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)
    for split_name, src in found.items():
        dst = os.path.join(TARGET_DIR, split_name)
        if os.path.exists(dst):
            print(f"\n'{dst}' já existe — pulando cópia.")
            continue
        print(f"\nCopiando {src} -> {dst} ...")
        shutil.copytree(src, dst)

    print(f"\n✅ Pronto. Estrutura final em '{TARGET_DIR}':")
    print_tree(TARGET_DIR, max_depth=2)


if __name__ == "__main__":
    main()