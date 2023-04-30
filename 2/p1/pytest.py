import numpy as np
import argparse
import yaml
from easydict import EasyDict

# a = {"a" : 1, "b": 2, "c": 3}
# print(list(a.keys()))
a = [2,3,5]
a = np.array(a)
print(a**2)
# b = np.array([[1,2,1],[2,1,2]])
# # print(type(a.sum()))
# # print(type(np.sum(a)))
# # for i, x in enumerate(a):
# #     x += 1
# c = a+b
# print(c)
# argc = np.argmax(c, axis=1)
# print(argc)
# print(c.shape[0])
# d = [c[i, argc[i]] for i in range(c.shape[0])]
# print(d)

# parser = argparse.ArgumentParser(description='HMM')
# parser.add_argument("--config_path", type=str, default="config.yaml")
# args = parser.parse_args()
# config_path =args.config_path
# config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
# config = EasyDict(config)
# config = config["HMM"]

# d = config["Train"]["Cn"]["tag_dict"]
# print(d["O"])

# a = np.array([1,0,2,0,4])
# print(a)
# a[a==0] = -1
# print(a)