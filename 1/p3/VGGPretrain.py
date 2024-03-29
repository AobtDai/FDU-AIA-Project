import torch
from torchvision import models
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from easydict import EasyDict


parser = argparse.ArgumentParser(description='VGG11 Pretrain')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["VGGPretrain"]

class_num = config["General"]["class_num"]
batch_size = config["Train"]["batch_size"]
epochs = config["Train"]["epochs"]
lr = config["Train"]["lr"]
is_load = config["Train"]["is_load"]
load_path = config["Train"]["load_path"]
save_path = config["Train"]["save_path"]


def eval(model, transform):
    # print(" -------< Evaluating >------- \n")
    eval_dataset = datasets.MNIST(root = 'data', 
                                  train = False, 
                                  transform = transform, 
                                  download = True)
    eval_loader = DataLoader(dataset = eval_dataset, 
                                              batch_size = 1, 
                                              shuffle = True)
    acc_num = 0
    for i, batch in enumerate(eval_loader):
        img_tensor, label_tensor = batch # torch.tensor
        with torch.no_grad():
            pred_tensor = model(img_tensor.to(device))
            for j in range(pred_tensor.size(0)):
                pred = torch.argmax(pred_tensor[j]).item()
                if pred==label_tensor[j]:
                    acc_num+=1

    acc_rate = acc_num / len(eval_dataset)
    # print("Acc_rate %.2f%% \n" % (acc_rate*100))
    return acc_rate


def build_model():
    vgg11 = models.vgg11(pretrained=True)
    vgg11.classifier[3] = nn.Linear(4096, 512)
    vgg11.classifier[6] = nn.Linear(512, class_num)
    return vgg11


if __name__ == "__main__":
    
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.to(device)

    if is_load:
        print(" -------< Loading parameters from {} >------- \n".format(load_path))
        # params = torch.load(load_path, map_location='cuda:0')
        params = torch.load(load_path, map_location="cpu")
        model.load_state_dict(params, strict=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.MNIST(root = 'data', 
                                train = True, 
                                transform = transform, 
                                download = True)
    train_loader = DataLoader(dataset = train_dataset, 
                            batch_size = batch_size, 
                            shuffle = True)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    loss_function = nn.CrossEntropyLoss()

    best_acc = 0.
    for epoch in range(1, epochs):
        for i, batch in enumerate(train_loader):
            img_tensor, label_tensor = batch
            # print(i)
            pred_tensor = model(img_tensor.to(device))
            loss = loss_function(pred_tensor, label_tensor.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_rate = eval(model, transform)
        print('Train epoch:[%d/%d]  loss:%.4f  acc:%.2f%%'
                % (epoch, epochs, loss, acc_rate*100))
        if acc_rate > best_acc:
            torch.save(model.state_dict(), save_path)
