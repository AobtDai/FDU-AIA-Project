r""" This file is to train and eval the classify-model """
r"""博1bo  学2xue  笃3du  志4zhi,
    切5qie 问6wen  近7jin 思8si, 
    自9zi  由10you 无11wu 用12yong """

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from CNNModel import CNNModel
from MyDataset import MyDataset
from torch.utils.data import DataLoader
import argparse
import yaml
from easydict import EasyDict


parser = argparse.ArgumentParser(description='Classifier Task With CNN')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["CNN"]

anno_train_path = config["Train"]["annotation_path"]
anno_val_path = config["Val"]["annotation_path"]
class_num = config["General"]["class_num"]
batch_size = config["Train"]["batch_size"]
epochs = config["Train"]["epochs"]
lr = config["Train"]["lr"]
is_load = config["Train"]["is_load"]
load_path = config["Train"]["load_path"]
save_path = config["Train"]["save_path"]


def eval(model):
    # model.eval()
    print(" -------< Evaluating >------- \n")
    eval_dataset = MyDataset(annotation_path = anno_val_path,
                              class_num = class_num, )
    eval_loader = DataLoader(dataset = eval_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = False)
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
    print("Acc_rate %.2f%% \n" % (acc_rate*100))
    return acc_rate


if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNNModel()
    model.to(device)

    if is_load:
        print(" -------< Loading parameters from {} >------- \n".format(load_path))
        params = torch.load(load_path, map_location='cuda:0')
        model.load_state_dict(params, strict=True) 

    loss_function = nn.CrossEntropyLoss() # including softmax
    optimizer = optim.Adam(model.parameters(), lr = lr)
    # optimizer = optim.RMSprop(model.parameters(), lr = lr)
    train_dataset = MyDataset(annotation_path = anno_train_path,
                              class_num = class_num, )
    train_loader = DataLoader(dataset = train_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = True)
    best_acc = 0.9839
    epoch_record_x = []
    acc_rate_record_y = []
    epoch_record_x.append(0)
    acc_rate_y = eval(model)
    acc_rate_record_y.append(acc_rate_y)

    for epoch in range(0, epochs+1):
        for i, batch in enumerate(train_loader):
            img_tensor, label_tensor = batch # torch.tensor
            pred_tensor = model(img_tensor.to(device))
            loss = loss_function(pred_tensor, label_tensor.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            print("Epoch" , epoch)
            epoch_record_x.append(epoch)
            acc_rate_y = eval(model)
            
            if best_acc < acc_rate_y:
                print(" -------< Saved Best Model >------- \n")
                torch.save(model.state_dict(), save_path)
                best_acc = acc_rate_y

            acc_rate_record_y.append(acc_rate_y)

            if epoch in [30,50]:
                lr = optimizer.param_groups[0]['lr']
                print(" -------< Reducing lr to {} >------- \n".format(lr/2))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr/2
        
    
    plt.plot(epoch_record_x, acc_rate_record_y, 
                color="green", label="acc_rate_record")
    plt.title("Loss with epoch")
    plt.legend()
    plt.show()
