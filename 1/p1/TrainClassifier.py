r""" This file is to train and eval the classify-model """
r"""博1bo  学2xue  笃3du  志4zhi,
    切5qie 问6wen  近7jin 思8si, 
    自9zi  由10you 无11wu 用12yong """


import math
import random
import matplotlib.pyplot as plt
import numpy as np
from ClassifierModel import ClassifierNet
from MyDataset import MyDataset
from torch.utils.data import DataLoader

anno_train_path = "C:/Users/25705/Downloads/documents/fdu/1.6/AI/PJ/1/p1/anno_store/anno_train.txt"
anno_val_path = "C:/Users/25705/Downloads/documents/fdu/1.6/AI/PJ/1/p1/anno_store/anno_val.txt"
class_num = 12
batch_size = 20

def eval(model, if_draw=False):
    eval_dataset = MyDataset(annotation_path = anno_val_path,
                              class_num = class_num, )
    eval_loader = DataLoader(dataset = eval_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = False)
    acc_num = 0
    total_loss = 0
    for i, batch in enumerate(eval_loader):
        img_tensor, label_tensor = batch # torch.tensor
        for j in range(0, min(len(label_tensor), batch_size)):
            # print(" >>>>>> ", img_tensor.shape, label_tensor.shape)
            img, label = img_tensor[j], label_tensor[j]
            img = img.numpy()
            pred = model.forward(img)
            pred_label = pred.argmax()+1
            if pred_label==label:
                acc_num+=1
            total_loss -= np.log(pred[label-1][0])
    acc_rate = acc_num / len(eval_dataset)
    avg_loss = total_loss / len(eval_dataset)
    print("eval_accuracy, %.2f total loss in %d data size" 
          % (total_loss, len(eval_dataset)))
    print("Avg_loss is %.2f, with acc_rate %.4f \n" % (avg_loss, acc_rate))
    return avg_loss, acc_rate


if __name__ == "__main__":
    BPClassifier = ClassifierNet(layer_arch = [28*28,64,64,12], 
                                 lr = 0.01, 
                                 batch_size = batch_size,
                                 task_kind="Classify")
    
    train_dataset = MyDataset(annotation_path = anno_train_path,
                              class_num = class_num, )
    
    train_loader = DataLoader(dataset = train_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = True)
    epoch_record_x = []
    avg_loss_record_y = []
    acc_rate_record_y = []
    epoch_record_x.append(0)
    avg_loss_y, acc_rate_y = eval(BPClassifier)
    avg_loss_record_y.append(avg_loss_y)
    acc_rate_record_y.append(acc_rate_y)

    for epoch in range(0,401):
        for i, batch in enumerate(train_loader):
            img_tensor, label_tensor = batch # torch.tensor
            for j in range(0, batch_size):
                img, label = img_tensor[j], label_tensor[j]
                img = img.numpy()
                pred = BPClassifier.forward(img)
                gt_one_hot = [0]*12
                gt_one_hot[label-1] = 1
                gt_one_hot = np.array(gt_one_hot).reshape((12,1))
                loss = pred - gt_one_hot
                BPClassifier.backward(loss)
            BPClassifier.update_weight(BPClassifier.lr)
        
        if epoch % 20 == 0:
            print("Epoch" , epoch)
            if_draw = False
            epoch_record_x.append(epoch)
            avg_loss_y, acc_rate_y = eval(BPClassifier)
            avg_loss_record_y.append(avg_loss_y)
            acc_rate_record_y.append(acc_rate_y)
    
    plt.plot(epoch_record_x, avg_loss_record_y,
                color="red", label="avg_loss_record")
    plt.plot(epoch_record_x, acc_rate_record_y, 
                color="green", label="acc_rate_record")
    plt.title("Loss with epoch")
    plt.legend()
    plt.show()
