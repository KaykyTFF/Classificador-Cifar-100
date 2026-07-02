# CIFAR-100 Classification

Projeto de classificação de imagens utilizando o dataset CIFAR-100. 

> **Desenvolvido por:** Kayky Terles Ferreira Feitosa  
> **Disciplina:** Inteligência Artificial  
> **Curso:** Análise e Desenvolvimento de Sistemas (ADS) - 3º Período  
> **Instituição:** IFPI - Campus Paulistana

## Estrutura do Projeto

- `src/model.py`: Definição da arquitetura da rede neural.
- `src/data_loader.py`: Carregamento e pré-processamento dos dados (CIFAR-100).
- `src/train.py`: Script de treinamento do modelo.
- `src/evaluate.py`: Script para avaliação do modelo treinado.
- `data/`: Diretório base para o dataset.
- `resultados/`: Métricas e gráficos de saída.
- `*.pth`: Pesos salvos do modelo e histórico (ex: `melhor_modelo_cifar100.pth`).

## Como Executar

### 1. Dependências
Instale os pacotes principais (ex: PyTorch, Torchvision, Matplotlib):
```bash
pip install torch torchvision matplotlib
```

### 2. Treinamento
Execute o script de treino:
```bash
python src/train.py
```

### 3. Avaliação
Para avaliar o modelo usando os pesos treinados:
```bash
python src/evaluate.py
```
