import os
import random

root_path = "D:/test_data"
# root_path->test file root path
anno_test_path = "C:/Users/25705/Downloads/documents/fdu/1.6/AI/PJ/1/p1/anno_store/anno_test.txt"

class_num = 12
num_in_class = 240 # ->new count of one class

test_file = open(anno_test_path, "w")
for i in range(1, class_num+1):
    for j in range(1, num_in_class+1):
        save_file = os.path.join(root_path, str(i), str(j)+".bmp")
        test_file.write(save_file + " %d \n" % i)