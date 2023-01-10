import argparse

import torch.optim
import torch.nn.functional as f

from loader import get_train_loader, get_val_loader
from tqdm import tqdm
from vit import VisionTransformer
arg_parser = argparse.ArgumentParser(description="My Implementation of Vision Transformer")
# Basic arguments for training
arg_parser.add_argument('data', metavar='DIR',
                        help="Place your training data and validation data together in one folder")
arg_parser.add_argument('epochs', type=int, default=90, help="Epochs for training")
# Basic arguments for optimization
arg_parser.add_argument('lr', type=float, default=0.1, help="Learning rate at start")
arg_parser.add_argument('momentum', type=float, default=0.9)
weight_decay = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def adjust_learning_rate(optimizer, epoch, args):
    """每隔30个Epoch就减小10倍学习率"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch(model, optimizer, criterion, train_loader):
    model.train()
    avg_loss = 0
    for idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        outputs = f.softmax(outputs, dim=1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        print(f"Batch {idx} loss {float(loss.cpu())}")
        optimizer.step()
        avg_loss += loss
    avg_loss /= len(train_loader)
    print(f"Train Loss: {avg_loss}")


def validation(model, criterion, val_loader):
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            outputs = f.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss
    avg_loss /= len(val_loader)
    print(f"Val Loss: {avg_loss}")


if __name__ == '__main__':
    args = arg_parser.parse_args()
    count_epochs = args.epochs  # 训练多少个Epoch
    data_dir = args.data  # 数据位置
    learning_rate = args.lr  # 学习率
    momentum = args.momentum  # 动量相关的设置
    model = VisionTransformer(cls_head=True, use_conv_stem=False, use_conv_stem_original=False, use_linear_patch=True)  # 召唤一个基础的Vision Transformer，但是要有分类头
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(count_epochs):
        print(f"Epoch number {epoch}")
        train_one_epoch(model, optimizer, criterion, train_loader=get_train_loader(data_dir + "/train"))
        # validation(model, criterion, get_val_loader(data_dir + "./val"))
