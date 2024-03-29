import numpy as np
import argparse
import yaml
from easydict import EasyDict
from collections import OrderedDict


parser = argparse.ArgumentParser(description='HMM')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["HMM"]
# kind = "Cn" #### 
kind = "En" #### 
train_path = config["Train"][kind]["train_path"]
val_path = config["Val"][kind]["val_path"]
out_path = config["Val"][kind]["out_path"]


def GetDict(path_lists):
    word_dict = OrderedDict()
    for path in path_lists:
        with open(path, "r", encoding="utf-8") as f:
            annotations = f.readlines()
        for annotation in annotations:
            splited_string = annotation.strip(" ").split(" ")
            if len(splited_string)<=1:
                continue
            word = splited_string[0]
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict


def GetData(path):
    tot_raw_words = [] 
    tot_raw_tags = []
    raw_words = []
    raw_tags = []
    with open(path, "r", encoding="utf-8") as f:
        annotations = f.readlines()
    for annotation in annotations:
        splited_string = annotation.strip(" ").strip("\n").split(" ")
        if len(splited_string)<=1:
            tot_raw_words.append(raw_words)
            tot_raw_tags.append(raw_tags)
            raw_tags = []
            raw_words = []
            continue
        word = splited_string[0]
        tag = splited_string[1]
        raw_words.append(word)
        raw_tags.append(tag)
    tot_raw_words.append(raw_words)
    tot_raw_tags.append(raw_tags)
    return tot_raw_words, tot_raw_tags


class HMMModel():
    # def __init__(self, word_dict, tag_dict, train_words, train_tags, val_words, val_tags):
    def __init__(self, word_dict, tag_dict, train_words, train_tags):
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.tot_words = train_words
        self.tot_tags = train_tags
        
        self.trans = np.zeros((len(tag_dict), len(tag_dict)))
        self.emits = np.zeros((len(tag_dict), len(word_dict)))
        self.piinit = np.zeros(len(tag_dict))

        for i, tags in enumerate(self.tot_tags):
            for j, tag in enumerate(tags):
                obser_word = self.tot_words[i][j]
                self.emits[tag_dict[tag]][word_dict[obser_word]] += 1
                if j==0:
                    self.piinit[tag_dict[tag]] += 1
                if j==len(tags)-1:
                    pass
                else :
                    next_tag = tags[j+1]
                    self.trans[tag_dict[tag]][tag_dict[next_tag]] += 1

        self.piinit = self.piinit / self.piinit.sum()
        # self.piinit = self.piinit**2 ###
        # self.piinit = self.piinit / self.piinit.sum()###
        self.piinit[self.piinit==0] = 1e-8
        self.piinit = np.log10(self.piinit)

        for i in range(0, len(self.trans)):
            sum = self.trans[i].sum()
            if sum==0:
                self.trans[i] = 0
            else :
                self.trans[i] = self.trans[i] / sum
                # self.trans[i] = self.trans[i]**2 ###
                # self.trans[i] = self.trans[i] / self.trans[i].sum() ###
            self.trans[i][self.trans[i]==0] = 1e-8
            self.trans[i] = np.log10(self.trans[i])
        
        for i in range(0, len(self.emits)):
            sum = self.emits[i].sum()
            if sum==0:
                self.emits[i] = 0
            else :
                self.emits[i] = self.emits[i] / sum
                # self.emits[i] = self.emits[i]**2 ###
                # self.emits[i] = self.emits[i] / self.emits[i].sum() ###
            self.emits[i][self.emits[i]==0] = 1e-8
            self.emits[i] = np.log10(self.emits[i])

    def val_fn(self, val_words, out_path):
        f = open(out_path, "w", encoding="utf-8")
        for i, words in enumerate(val_words):
            prob = np.zeros((len(words), len(self.tag_dict)))
            arg_max_p = np.zeros((len(words), len(self.tag_dict)))
            states = np.zeros(len(words))    
            prob[0] = self.piinit + self.emits[:, self.word_dict[words[0]]]
            for j in range(1, len(words)):
                max_p = prob[j-1] + self.trans.T
                arg_max_p[j] = np.argmax(max_p, axis=1)
                # print(type(arg_max_p[j][0]))
                prob[j] = [max_p[k, int(arg_max_p[j][k])] for k in range(max_p.shape[0])]
                prob[j] = prob[j] + self.emits[:, self.word_dict[words[j]]]
            
            states[-1] = np.argmax(prob[-1])
            for j in reversed(range(0, len(prob)-1)):
                states[j] = arg_max_p[j+1][int(states[j+1])]
                # cannot argmax, or it will get trapped in local best
            rev_tag_dict = list(self.tag_dict.keys())
            for j in range(len(states)): # find the key
                f.write(words[j] + " " + rev_tag_dict[int(states[j])] + "\n")
            if i!=len(val_words)-1:
                f.write("\n")

        f.close()



if __name__ == "__main__":

    word_dict = GetDict([train_path, val_path])
    tag_dict = config["Train"][kind]["tag_dict"]

    train_words, train_tags = GetData(train_path)
    val_words, val_tags = GetData(val_path)

    # HMM = HMMModel(word_dict, tag_dict, train_words, train_tags, val_words, val_tags)
    HMM = HMMModel(word_dict, tag_dict, train_words, train_tags)
    HMM.val_fn(val_words, out_path)



    

    
    

