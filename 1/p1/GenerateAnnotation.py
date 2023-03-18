r""" this file is to generate both train and validation dataset"""

import os
import random

root_path = "C:/Users/25705/Downloads/documents/fdu/1.6/AI/PJ/1/p1/train_data"
anno_train_path = "C:/Users/25705/Downloads/documents/fdu/1.6/AI/PJ/1/p1/anno_store/anno_train.txt"
anno_val_path = "C:/Users/25705/Downloads/documents/fdu/1.6/AI/PJ/1/p1/anno_store/anno_val.txt"

class_num = 12
num_in_class = 620

train_file = open(anno_train_path, "w")
val_file = open(anno_val_path, "w")
for i in range(1, class_num+1):
    val_num = 0
    for j in range(1, num_in_class+1):
        save_file = os.path.join(root_path, str(i), str(j)+".bmp")
        randamx = random.randint(1, 100)
        if val_num < num_in_class//10 and randamx < 30: # make it a val
            val_file.write(save_file+ " %d \n" % i)
            val_num += 1
        else:
            train_file.write(save_file+ " %d \n" % i)
