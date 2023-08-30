import torch
import torch.nn as nn
import pandas as pd
from tqdm.notebook import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt

def init_batch_generator(dataloader: torch.utils.data.DataLoader):
    """
    Возвращает функцию, вызов которой возвращает следующие batch_size
    примеров и им соответствуюющих меток из train_data, train_labels.
    
    Примеры выбираются последовательно, по кругу. Массивы с входными 
    примерами и метками классов перемешиваются в начале каждого круга.
    """
    def f():
        while True:
            for i, (images, labels) in enumerate(dataloader):
                yield images, labels
    return f()

def train(onn: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler,
          device: torch.device = torch.device('cpu'),
          test_loader: torch.utils.data.DataLoader = None,
          num_epochs: int = 5,
          func_transform = None,
          get_train_data: bool = False,
          loss_list: torch.Tensor | None = None,
          acc_list: torch.Tensor | None = None):
    
    acc_test_list = None

    divider: int
    if get_train_data:
        divider = int(num_epochs*len(train_loader)/len(loss_list))
    
    if test_loader is not None:
        acc_test_list = torch.zeros(int(num_epochs*len(train_loader)/divider))
    

    onn.train()

    data_iterator = init_batch_generator(train_loader)
    test_iterator = init_batch_generator(test_loader)
    progress = trange(num_epochs*len(train_loader))

    for epoch in progress:
        onn.train()
        # Прямой запуск
        images, labels = next(data_iterator)
        images = images.to(device)
        labels = labels.to(device)
        #labels =  torch.tensor([[ 1. if j == labels[i] else 0. for j in range(10)]for i in range(len(labels))]).to(device)
        _, outputs = onn(images)
        loss = criterion(outputs, labels if func_transform is None else func_transform(labels))
        # if get_train_data and (epoch + 1) % divider == 0:
        #     loss_list[int(epoch/divider)] = loss.item()

        # Обратное распространение и оптимизатор
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Отслеживание точности на тренировочном наборе
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        if get_train_data and (epoch + 1) % divider == 0:
            onn.eval()
            _, outputs = onn(images)
            loss_list[int(epoch/divider)] = criterion(outputs, labels if func_transform is None else func_transform(labels)).item()
            #total = labels.size(0)
            _, predicted2 = torch.max(outputs.data, 1)
            acc_list[int(epoch/divider)] = (predicted2 == labels).sum().item() / total

        # Отслеживание точности на тестовом наборе
        if test_loader is not None:
            images, labels = next(test_iterator)
            images = images.to(device)
            labels = labels.to(device)
            _, outputs = onn(images)
            total_test = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_test = (predicted == labels).sum().item()
            acc_test_list[epoch] = correct_test / total_test

        if (epoch + 1) % 5 == 0:
            scheduler.step()
        if (epoch+1) % 7 == 0:
            if test_loader is not None:
                progress.set_postfix_str(f"Loss: {loss.item() :.4f}, Accuracy: {(correct / total) * 100 :.2f}%, Test accuracy: {acc_test_list[epoch]*100 :.2f} lr: {scheduler.get_last_lr()[0] :e}")
            else:
                progress.set_postfix_str(f"Loss: {loss.item() :.4f}, Accuracy: {(correct / total) * 100 :.2f}%, lr: {scheduler.get_last_lr()[0] :e}")
    if test_loader is not None:
        return acc_test_list

def test(onn: nn.Module,
         test_dataloader: torch.utils.data.DataLoader,
         device: torch.device = torch.device('cpu'),
         get_acc_df: bool = False,
         get_energy_df: bool = False):
         
    pd.options.styler.format.precision = 1

    df_energy = df_acc = energy_max = energy = acc_max = acc = None
    
    if get_acc_df:
        acc = torch.zeros((10, 10)).to(device)
        acc_max = torch.zeros((10, 1)).to(device)

    if get_energy_df:
        energy = torch.zeros((10, 10)).to(device)
        energy_max = torch.zeros((10, 1)).to(device)

    onn.eval()

    data_iterator = init_batch_generator(test_dataloader)
    progress = trange(len(test_dataloader))

    with torch.no_grad():
        correct = 0
        total = 0
        for i in progress:
            images, labels = next(data_iterator)
            out_image, outputs = onn(images.to(device))

            if get_energy_df:
                out_image=out_image*onn.mask
                out_energy = torch.sum(torch.abs(out_image)**2, dim = (2, 3))
                for i in range(len(labels)):
                    for j in range(10):
                        energy[labels[i]][j]+=out_energy[i][j]
                        energy_max[labels[i]][0]+=out_energy[i][j]

            _, predicted = torch.max(outputs.data, 1)

            if get_acc_df:
                for i in range(len(predicted)):
                    acc[predicted[i]][labels[i]]+=1
                    acc_max[labels[i]][0]+=1
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            progress.set_postfix_str(f"Accuracy: {(correct / total) * 100}%")

    if get_acc_df:
        acc/=acc_max/100
        df_acc = pd.DataFrame(acc.cpu(),
                                index=pd.MultiIndex.from_product([['Точность предсказания'], range(10)]),
                                columns=pd.MultiIndex.from_product([['Поданное число'], range(10)]),).style.background_gradient(cmap='YlOrBr').set_table_styles([
            {'selector': 'th.col_heading', 'props': 'text-align: center;'},
            {'selector': 'th.row_heading.level0', 'props': 'writing-mode: vertical-lr; transform: rotate(180deg); text-align: center;'},
        ], overwrite=False)
    
    if get_energy_df:
        energy/=energy_max/100
        df_energy = pd.DataFrame(torch.transpose(energy, 0, 1).cpu(),
                                index=pd.MultiIndex.from_product([['Расперделение энергии'], range(10)]),
                                columns=pd.MultiIndex.from_product([['Поданное число'], range(10)])).style.background_gradient().set_table_styles([
            {'selector': 'th.col_heading', 'props': 'text-align: center;'},
            {'selector': 'th.row_heading.level0', 'props': 'writing-mode: vertical-lr; transform: rotate(180deg); text-align: center;'},
        ], overwrite=False)
    
    if get_energy_df and not get_acc_df:
        return df_energy
    else:
        return df_acc, df_energy
    
def show_image(image: torch.Tensor,
               number_image: torch.Tensor,
               number: int = 0,
               acc: torch.Tensor | None = None,
               showing_grid: bool = False,
               coords: torch.Tensor | None = None,
               image_size: int = 56,
               fig_size: int = 5):
    fig = plt.figure(figsize=(fig_size*2, fig_size))
    ax = fig.add_subplot(1, 2, 1)
    for i in range(10):
        if showing_grid:
            rectangle = plt.Rectangle((coords[i]*100+482)/2,
                               image_size/2, image_size/2, fc='#00000000', ec="black")
            ax.add_patch(rectangle)
        if acc is not None:
            ax.text((coords[i][0]*100+476)/2, (coords[i][1]*100+598)/2, f"{acc[number][i].item()*100 :.1f}%", fontdict={ 'color': 'black' })
    plt.colorbar(ax.imshow(np.squeeze(torch.abs(image[number]).cpu().detach().numpy()**2), cmap='binary'))
    ax = fig.add_subplot(1, 2, 2)
    plt.colorbar(ax.imshow(np.squeeze(number_image[number].cpu().detach().numpy()), cmap='binary'))