import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from textrnn.model import LSTM
from textrnn.dataloader import MyDataset
from textrnn.config import RECORD_PATH

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def train(train_loader, model, optimizer, loss_func, device):
    model.train()
    y_true, y_pred, loss_mean = [], [], 0.0
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        loss_mean += loss.data
        _, pred = output.data.max(1)
        y_true += target.to('cpu').data.numpy().tolist()
        y_pred += pred.to('cpu').data.numpy().tolist()
    loss_mean = loss_mean / len(train_loader)
    return y_pred, y_true, loss_mean


def eval(test_loader, model, device):
    model.eval()
    y_true, y_pred = [], []
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        _, pred = output.data.max(1)
        y_true += target.to('cpu').data.numpy().tolist()
        y_pred += pred.to('cpu').data.numpy().tolist()
    return y_pred, y_true


def load_model(model, epoch):
    if epoch != 0:
        model.load_state_dict(torch.load(os.path.join(RECORD_PATH, get_model_name(model, epoch))))
    return model


def get_model_name(model, epoch):
    return "{}.{}.pth".format(model._get_name(), epoch)


if __name__ == '__main__':
    seq_len = 380
    start_epoch = 0
    max_epoch = 1000
    lr = 0.005
    batch_size = 512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("加载数据")
    train_loader = DataLoader(MyDataset('train', seq_len), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MyDataset('test', seq_len), batch_size=batch_size)

    print("加载模型：epoch:{}".format(start_epoch))
    print("训练方式：{}".format(device))
    model = LSTM()
    model = load_model(model, start_epoch)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    print("开始训练：{}".format(model._get_name()))
    max_test_acc = 0
    for epoch in range(start_epoch, max_epoch):
        epoch += 1
        start = time.time()
        y_true, y_pred, loss_mean = train(train_loader, model, optimizer, loss_func, device)
        train_acc = accuracy_score(y_true, y_pred)
        y_true, y_pred, = eval(test_loader, model, device)
        test_acc = accuracy_score(y_true, y_pred)
        span = time.time() - start
        print("After %s epoch, || cost=%.2fs | loss=%.4f | acc=[train=%.2f%%, test=%.2f%%] %s" % (
            epoch, span, loss_mean, train_acc * 100, test_acc * 100, " TOP!" if max_test_acc < test_acc else ""))
        if max_test_acc < test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(RECORD_PATH, get_model_name(model, epoch)))
