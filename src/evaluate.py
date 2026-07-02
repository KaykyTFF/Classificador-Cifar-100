import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np
from colorama import Fore, Style, init

from data_loader import get_dataloaders
from model import CIFAR100_CNN

# Inicializar colorama para o Windows
init(autoreset=True)

def evaluate_model():
    # Criar pasta para salvar resultados
    output_dir = 'resultados'
    os.makedirs(output_dir, exist_ok=True)

    # Configurando o dispositivo (CUDA, DirectML para AMD ou CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA (GPU)"
    else:
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
                device_name = "DirectML (GPU AMD/Intel/NVIDIA)"
            else:
                device = torch.device("cpu")
                device_name = "CPU"
        except ImportError:
            device = torch.device("cpu")
            device_name = "CPU"
            
    print(f"{Fore.CYAN}Dispositivo utilizado: {device} ({device_name}){Style.RESET_ALL}")


    print(f"{Fore.YELLOW}Carregando dataloader de teste...{Style.RESET_ALL}")
    _, testloader, classes = get_dataloaders(batch_size=100, num_workers=2)

    # Inicializa a arquitetura e carrega os pesos
    net = CIFAR100_CNN().to(device)
    
    if not os.path.exists('melhor_modelo_cifar100.pth'):
        print(f"{Fore.RED}Erro: Arquivo 'melhor_modelo_cifar100.pth' não encontrado. Execute o treinamento primeiro.{Style.RESET_ALL}")
        return
        
    print(f"{Fore.YELLOW}Carregando os pesos do melhor modelo salvo...{Style.RESET_ALL}")
    net.load_state_dict(torch.load('melhor_modelo_cifar100.pth', weights_only=False))
    
    print(f"{Fore.GREEN}Gerando gráficos e avaliações...{Style.RESET_ALL}")

    # Carregar o histórico do treinamento
    if os.path.exists('historico_treinamento.pth'):
        historico = torch.load('historico_treinamento.pth', weights_only=False)
        train_losses = historico['train_losses']
        test_losses = historico['test_losses']
        train_accs = historico['train_accs']
        test_accs = historico['test_accs']
        
        # Plot Treinamento (Loss e Accuracy)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Treino', color='blue')
        plt.plot(test_losses, label='Teste', color='orange')
        plt.title('Gráfico de Treinamento - Perda (Loss)')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Treino', color='blue')
        plt.plot(test_accs, label='Teste', color='orange')
        plt.title('Gráfico de Treinamento - Acurácia')
        plt.xlabel('Época')
        plt.ylabel('Acurácia (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grafico_treinamento.png'))
        plt.close()
        print(f"{Fore.CYAN}Gráfico salvo: '{output_dir}/grafico_treinamento.png'{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Aviso: 'historico_treinamento.pth' não encontrado. Gráfico de histórico não será gerado.{Style.RESET_ALL}")

    # Avaliação Completa (Matriz e Top-5 Accuracy)
    net.eval()
    all_preds = []
    all_labels = []
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    print(f"{Fore.YELLOW}Avaliando os dados de teste para métricas detalhadas...{Style.RESET_ALL}")
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            
            # Top-1
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Top-5 Accuracy
            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            correct_k = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))
            
            correct_top1 += correct_k[:1].reshape(-1).float().sum(0, keepdim=True).item()
            correct_top5 += correct_k[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total += labels.size(0)

    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total
    print(f"\n{Fore.GREEN}=== RESULTADOS FINAIS ==={Style.RESET_ALL}")
    print(f"Top-1 Acurácia: {top1_acc:.2f}% (A rede acerta a classe exata)")
    print(f"Top-5 Acurácia: {top5_acc:.2f}% (A classe correta está entre as 5 primeiras opções)")

    # Matriz de Confusão
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
    plt.title("Matriz de Confusão (CIFAR-100)", fontsize=16)
    plt.xlabel("Classe Predita", fontsize=14)
    plt.ylabel("Classe Verdadeira", fontsize=14)
    plt.savefig(os.path.join(output_dir, 'matriz_confusao.png'))
    plt.close()
    print(f"{Fore.CYAN}Matriz salva: '{output_dir}/matriz_confusao.png'{Style.RESET_ALL}")

    # Acurácia por Classe (Melhores e Piores)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    acc_dict = {classes[i]: class_acc[i] for i in range(100)}
    
    # Ordenar as classes pela acurácia
    sorted_acc = sorted(acc_dict.items(), key=lambda x: x[1])
    piores = sorted_acc[:10]
    melhores = sorted_acc[-10:]
    
    # Plotar Top 10 Melhores e Piores
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico Piores
    axes[0].barh([x[0] for x in piores], [x[1]*100 for x in piores], color='#e74c3c')
    axes[0].set_title('Top 10 Classes com PIOR Acurácia')
    axes[0].set_xlabel('Acurácia (%)')
    axes[0].set_xlim(0, 100)
    
    # Gráfico Melhores
    axes[1].barh([x[0] for x in melhores], [x[1]*100 for x in melhores], color='#2ecc71')
    axes[1].set_title('Top 10 Classes com MELHOR Acurácia')
    axes[1].set_xlabel('Acurácia (%)')
    axes[1].set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'melhores_piores_classes.png'))
    plt.close()
    print(f"{Fore.CYAN}Gráfico salvo: '{output_dir}/melhores_piores_classes.png'{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}Avaliação finalizada com sucesso! Todos os arquivos estão na pasta '{output_dir}'.{Style.RESET_ALL}")

if __name__ == '__main__':
    evaluate_model()
