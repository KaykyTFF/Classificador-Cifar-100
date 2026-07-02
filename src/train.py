import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from colorama import Fore, Style, init

from data_loader import get_dataloaders
from model import CIFAR100_CNN

# Inicializar colorama para o Windows
init(autoreset=True)

import numpy as np

# --- Funções para Mixup e Cutmix ---
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# -----------------------------------

# Função personalizada para criar uma barra de progresso limpa (não inunda o terminal do Kaggle)
def print_progress_bar(iteration, total, prefix='', length=40):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    # O \r substitui a linha atual em vez de criar dezenas de linhas novas
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% ')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def train_model():
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


    print(f"{Fore.YELLOW}Baixando e carregando CIFAR-100...{Style.RESET_ALL}")
    trainloader, testloader, classes = get_dataloaders(batch_size=128, num_workers=2)

    net = CIFAR100_CNN().to(device)

    # NLLLoss porque o modelo tem a camada Softmax na saída
    criterion = nn.NLLLoss()
    
    # Otimizador Clássico: SGD com Momentum (Weight Decay reduzido para 1e-4)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
    epochs = 400 # Aumentando épocas para dar mais tempo de aprendizado
    patience = 50 # Early Stopping de 50 épocas sem melhoria na validação
    
    # Scheduler Cosine Annealing estendido para as novas épocas com eta_min
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_loss = float('inf')
    patience_counter = 0

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    print(f"{Fore.GREEN}Iniciando treinamento da ResNet-34...{Style.RESET_ALL}")
    start_time = time.time()
    
    steps_per_epoch = len(trainloader)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        prefix_treino = f"{Fore.BLUE}Época [{epoch+1}/{epochs}] - Treino{Style.RESET_ALL}"
        
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Escolhe aleatoriamente entre Normal, Mixup ou Cutmix (33% de chance cada)
            r = np.random.rand(1)
            use_mix = False
            if r < 0.33:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, 1.0, device.type == 'cuda')
                use_mix = True
            elif r < 0.66:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, 1.0, device.type == 'cuda')
                use_mix = True
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = net(inputs)
                    if use_mix:
                        loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(inputs)
                if use_mix:
                    loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress bar limpa (sem criar centenas de linhas no Kaggle)
            print_progress_bar(i + 1, steps_per_epoch, prefix=prefix_treino)
            
        epoch_train_loss = running_loss / len(trainloader)
        epoch_train_acc = 100 * correct / total
        
        # Step no CosineAnnealing por época
        scheduler.step()
        
        # Avaliação
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        prefix_teste = f"{Fore.MAGENTA}Época [{epoch+1}/{epochs}] - Validação{Style.RESET_ALL}"
        
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                print_progress_bar(i + 1, len(testloader), prefix=prefix_teste)
                
        epoch_test_loss = test_loss / len(testloader)
        epoch_test_acc = 100 * correct / total
        
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(epoch_test_acc)
        
        print(f"Resultado Final da Época [{epoch+1}/{epochs}] | "
              f"Train Loss: {Fore.RED}{epoch_train_loss:.4f}{Style.RESET_ALL} | "
              f"Train Acc: {Fore.GREEN}{epoch_train_acc:.2f}%{Style.RESET_ALL} | "
              f"Test Loss: {Fore.RED}{epoch_test_loss:.4f}{Style.RESET_ALL} | "
              f"Test Acc: {Fore.GREEN}{epoch_test_acc:.2f}%{Style.RESET_ALL}\n")
        
        # Early Stopping Check
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            patience_counter = 0
            torch.save(net.state_dict(), 'melhor_modelo_cifar100.pth')
            print(f"{Fore.YELLOW}--> Novo melhor modelo salvo! (Loss: {best_loss:.4f}){Style.RESET_ALL}\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"{Fore.RED}Early Stopping ativado na época {epoch+1}!{Style.RESET_ALL}")
                break

    tempo_total = (time.time() - start_time)/60
    print(f"\n{Fore.GREEN}Treinamento finalizado em {tempo_total:.2f} minutos.{Style.RESET_ALL}")

    torch.save({
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }, 'historico_treinamento.pth')
    print(f"{Fore.CYAN}Histórico de métricas salvo para avaliação posterior.{Style.RESET_ALL}")

if __name__ == '__main__':
    train_model()
