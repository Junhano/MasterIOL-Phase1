import torch
import torch.nn as nn
import torch.cuda
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os


def train_stage1(epochs, learning_rate, batch_size, loss, checkpoint = None, save_path = None, device = 'cpu', cross_validate = False):
    def train(model: nn.Module, dataset, call_validate_func, validate_func, validate_dataset):
        print(f'Training using stage 1 method on model type {type(model)}')

        train_model = model.to(device)
        optimizer = optim.Adam(train_model.parameters(), lr=learning_rate)

        train_data_loader = DataLoader(dataset, batch_size=batch_size)

        train_model.train()
        train_scaler = GradScaler()
        train_loss_tracker = []
        if checkpoint:
            if save_path != None and os.path.exists(save_path):
                state = torch.load(save_path, map_location=device)
                train_model.load_state_dict(state['model_state_dict'])
            else:
                print('Warning, the path you provided doesn\'t exist')

        for epoch in range(epochs):
            loss_track = []
            for _, data in enumerate(train_data_loader):

                img = data[0].to(device)
                label = data[1].to(device)
                label = label[:, None]
                with autocast():
                    output = train_model(img)
                    train_loss = loss(output, label.float())
                    loss_track.append(train_loss.item())
                    optimizer.zero_grad()
                    train_scaler.scale(train_loss).backward()
                    train_scaler.step(optimizer)
                    train_scaler.update()

            train_loss_tracker.append(sum(loss_track) / len(loss_track))
            _, validation_accuracy, _, _,_,_ = call_validate_func(train_model, validate_func, None, validate_dataset, eval = False)
            print('Training Loss {} epochs {}, Validation accuracy is '.format(sum(loss_track) / len(loss_track), epoch + 1), validation_accuracy)

        if not cross_validate:
            if not save_path:
                save_path = './stage1_checkpoint.pt'

            torch.save({
                'model_state_dict': train_model.state_dict()
            }, save_path)
        return 1, train_loss_tracker

    return train