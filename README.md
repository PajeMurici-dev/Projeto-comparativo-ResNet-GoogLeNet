# Comparação ResNet-50 vs. GoogLeNet (Inception v1) — Detecção de Faces Reais vs. Falsas (Deepfakes)

Projeto para o artigo IEEE comparando as duas arquiteturas via *transfer
learning* com pesos pré-treinados na ImageNet, aplicado ao dataset
"140k Real and Fake Faces".

## 1. Estrutura do projeto

```
tumor_cnn_comparison/
├── config.py           # caminhos e hiperparâmetros — ajuste aqui primeiro
├── download_data.py     # baixa e organiza o dataset via kagglehub
├── dataset.py             # carregamento via ImageFolder (train/valid/test)
├── models.py                 # construção da ResNet-50 e do GoogLeNet
├── train.py                     # loop de treino/validação/teste + métricas
├── compare_models.py               # script principal: roda tudo e gera a comparação
└── requirements.txt
```

## 2. Sobre o dataset (140k Real and Fake Faces)

- **Tarefa**: classificação binária — a imagem é um rosto real (fotografado,
  do Flickr) ou um rosto gerado artificialmente (GAN)?
- **Classes**: `real` e `fake`.
- **Tamanho**: 70.000 imagens reais + 70.000 imagens falsas = 140.000 no
  total, já **perfeitamente balanceado** entre as classes.
- **Estrutura**: diferente do HAM10000, este dataset já vem organizado em
  pastas por classe e já vem dividido em `train/`, `valid/` e `test/`
  (formato `ImageFolder` do torchvision), então não é necessário fazer
  split manual nem lidar com um CSV de metadados.

Como o dataset é grande (~4GB, 140k imagens), o `config.py` tem um parâmetro
`SUBSET_FRACTION` para treinar com uma fração aleatória de cada split — útil
se você estiver rodando em ambiente com pouca GPU/tempo (ex: Colab
gratuito). Por padrão está em `1.0` (dataset completo).

## 3. Baixando o dataset

### Opção A — via kagglehub (recomendado)

```bash
pip install kagglehub
python download_data.py
```

Esse script baixa o dataset, localiza automaticamente as pastas
`train`/`valid`/`test` (verificando que cada uma contém subpastas `real` e
`fake`) e organiza tudo em `./data/train`, `./data/valid`, `./data/test`.

### Opção B — download manual (zip)

Baixe pelo site do Kaggle e extraia mantendo a mesma estrutura final:

```
data/
├── train/
│   ├── real/
│   └── fake/
├── valid/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## 4. Instalação

```bash
pip install -r requirements.txt
```

## 5. Rodando o experimento

```bash
python compare_models.py
```

Isso vai:
1. Carregar os três splits via `ImageFolder`;
2. (Opcionalmente) reduzir cada split conforme `SUBSET_FRACTION`;
3. Treinar ResNet-50 e GoogLeNet (transfer learning);
4. Avaliar os dois no conjunto de teste;
5. Gerar tabela comparativa e gráfico de curvas de acurácia.

## 6. Saídas geradas (pasta `results/`)

- `ResNet50_results.json` / `GoogLeNet_results.json`: métricas completas
  (acurácia, F1 macro, matriz de confusão, relatório por classe, tempo de
  treino, tempo de inferência, número de parâmetros).
- `comparison_table.txt`: tabela resumida para colar na seção de
  Resultados do artigo.
- `val_accuracy_comparison.png`: gráfico comparando a curva de acurácia
  de validação por época.
- `all_results.json`: os dois resultados juntos.

Checkpoints do melhor modelo de cada arquitetura (por F1 de validação)
ficam em `checkpoints/`.

## 7. Ajustes importantes para o artigo

- **`SUBSET_FRACTION`**: se o treino completo (140k imagens × 2 modelos)
  for inviável no seu ambiente, comece com algo como `0.1` ou `0.2` para
  validar que o pipeline funciona antes de rodar o experimento completo —
  vale mencionar essa limitação/decisão na seção de Metodologia se você
  usar um subconjunto no experimento final.
- **`FREEZE_BACKBONE`**: comece com `True`. Compare depois com `False`
  (fine-tuning completo) se quiser uma subseção extra de discussão —
  nesse dataset em particular, artefatos sutis de GAN podem exigir mais
  fine-tuning das camadas convolucionais do que um dataset com diferenças
  mais "macro" entre classes.
- Como as classes já estão balanceadas, a acurácia simples já é uma
  métrica confiável aqui (diferente do HAM10000) — mas ainda vale reportar
  F1 e a matriz de confusão para ver se algum modelo tende a confundir
  faces reais com falsas em alguma direção específica (falso positivo vs.
  falso negativo importam de forma diferente em aplicações reais de
  detecção de deepfake).

## 8. Notas sobre as arquiteturas

- **ResNet-50**: 50 camadas com peso, organizadas em blocos residuais
  (*bottleneck blocks*). A saída de cada bloco é `F(x) + x` em vez de só
  `F(x)` — as conexões residuais (skip connections) permitem treinar redes
  muito mais profundas sem sofrer tanto com vanishing gradient, já que o
  gradiente pode fluir diretamente pelos atalhos durante a
  backpropagation. ~25.6M de parâmetros.
- **GoogLeNet (Inception v1)**: 22 camadas com peso, usando módulos
  Inception que aplicam convoluções de tamanhos diferentes (1×1, 3×3, 5×5)
  em paralelo, concatenando as saídas. ~6.8M de parâmetros — bem mais
  leve que a ResNet-50. Usa duas saídas auxiliares durante o treino para
  ajudar a propagar o gradiente numa rede profunda.

Essa comparação rende uma discussão interessante pro artigo: detecção de
deepfake muitas vezes depende de artefatos visuais bem sutis (texturas,
inconsistências de iluminação, padrões de compressão) — vale discutir se
a maior capacidade representacional da ResNet-50 (mais parâmetros, blocos
residuais mais profundos) traduz-se em vantagem real de acurácia aqui, ou
se o GoogLeNet, mais enxuto, já captura esses padrões igualmente bem com
uma fração do custo computacional.