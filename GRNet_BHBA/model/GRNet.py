import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from metrics.plot_metrics import plot_acc_graph

from model.resi_block_main import ResidualBlockMain

class GRNet(nn.Module):
    def __init__(self, len_vec, num_classes):
        super(GRNet, self).__init__()
        fc_num = 64
        self.len_vec = int(len_vec)
        self.fc = nn.Linear(self.len_vec, self.len_vec, bias=False)
        
        self.norm = nn.LayerNorm(self.len_vec)
        self.fc64 = nn.Linear(self.len_vec, 64, bias=False)
        self.sftmx = nn.Softmax()
        self.fc5 = nn.Linear(64, 5, bias=False)
        self.res_main = ResidualBlockMain(self.len_vec)

    def forward(self, x):
        x = self.fc(x)
        x = self.res_main(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.fc64(x)
        x = self.sftmx(x)
        x = self.fc5(x)
        return x

    def train_model(self, epochs, train_loader, val_loader, model, classes):
        self.cri = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9,
                            weight_decay = 0.0005, nesterov=True)
        # self.opt = torch.optim.Adam(model.parameters(), lr=0.0001)

        train_acc = Accuracy(task="multiclass", num_classes=len(classes))
        valid_acc = Accuracy(task="multiclass", num_classes=len(classes))

        self.train_losses = []
        self.validation_losses = []
        self.total_train_acc = []
        self.total_validation_acc = []

        for i in range(epochs):
            for b, (X_train, y_train) in enumerate(train_loader):
                X_train = X_train.unsqueeze(1)
                max_val = torch.max(X_train)
                X_train = X_train*255/max_val
                y_pred_train = model(X_train)
                y_pred_train = y_pred_train.squeeze(1)
                loss_train = self.cri(y_pred_train, y_train)
                batch_tr = train_acc(y_pred_train, y_train)

                self.opt.zero_grad()
                loss_train.backward()
                self.opt.step()
            with torch.no_grad():
                for b, (X_val, y_val) in enumerate(val_loader):
                    X_val = X_val.unsqueeze(1)
                    max_val = torch.max(X_val)
                    X_val = X_val * 255 / max_val
                    y_pred_val = model(X_val)
                    y_pred_val = y_pred_val.squeeze(1)
                    loss_val = self.cri(y_pred_val, y_val)
                    batch_val = valid_acc(y_pred_val, y_val)

            self.validation_losses.append(loss_val)
            self.train_losses.append(loss_train)

            self.total_train_acc.append(train_acc.compute()*100)
            self.total_validation_acc.append(valid_acc.compute()*100)

            if i % 10 == 0:
                print(f'Epoch: {i} | Train loss: {self.train_losses[i]:10.8f} |'
                      f' Train accuracy: {self.total_train_acc[i]:.5}% |'
                      f' Val loss: {self.validation_losses[i]:10.8f} |'
                      f' Val accuracy: {self.total_validation_acc[i]:.5}%')

            train_acc.reset()
            valid_acc.reset()

        plot_acc_graph(self.train_losses, self.total_train_acc, self.validation_losses,
                       self.total_validation_acc, epochs)

    def test_model(self, test_loader, model):
        with torch.no_grad():
            correct_all = 0
            for b, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test.unsqueeze(1)
                max_val = torch.max(X_test)
                X_test = X_test * 255 / max_val
                y_pred_test = model(X_test)
                y_pred_test = y_pred_test.squeeze(1)
                y_predicted_test = torch.max(y_pred_test, 1)[1]
                correct_all += (y_predicted_test == y_test).sum().item()

            tst_loss = self.cri(y_pred_test, y_test)
            tst_acc = correct_all * 100 / int(X_test.shape[0])
        print(f'Test accuracy: {correct_all}/{int(X_test.shape[0])} = {tst_acc:7.3f}%')
        print(f'Test loss: { tst_loss:10.8f}')
        return y_test, y_predicted_test

