import numpy as np
import argparse
import yaml
from easydict import EasyDict
from collections import OrderedDict
from sklearn_crfsuite import CRF


parser = argparse.ArgumentParser(description='CRF')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["CRF"]
kind = "Cn" #### 
# kind = "En" #### 
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

def Word2Features(sent, i):
    word = sent[i]
    prev_word = '<s>' if i == 0 else sent[i-1] # START_TAG
    next_word = '</s>' if i == (len(sent)-1) else sent[i+1] # STOP_TAG
    prev_word2 = '<s>' if i <= 1 else sent[i-2] # START_TAG
    next_word2 = '</s>' if i >= (len(sent)-2) else sent[i+2] # STOP_TAG
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word + word,
        'w:w+1': word + next_word,
        'w-1:w:w+1': prev_word + word + next_word, # add
        'w-2:w': prev_word2 + word, # add
        'w:w+2': word + next_word2, # add
        'bias': 1
    }
    return features


def Sent2Features(sent):
    return [Word2Features(sent, i) for i in range(len(sent))]


class CRFModel(object):
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, 
                 max_iterations=100, all_possible_transitions=False):
        self.crf = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, train_words, train_tags):
        features = [Sent2Features(s) for s in train_words]
        self.crf.fit(features, train_tags)

    def val(self, val_words, word_dict, tag_dict, out_path):
        f = open(out_path, "w", encoding="utf-8")
        features = [Sent2Features(s) for s in val_words]
        preds = self.crf.predict(features)
        for i, words in enumerate(val_words):
            for j in range(len(words)): # find the key
                f.write(words[j] + " " + preds[i][j] + "\n")
            if i!=len(val_words)-1:
                f.write("\n")
        f.close()


if __name__ == "__main__":

    word_dict = GetDict([train_path, val_path])
    tag_dict = config["Train"][kind]["tag_dict"]

    train_words, train_tags = GetData(train_path)
    val_words, val_tags = GetData(val_path)

    crf = CRFModel()
    crf.train(train_words, train_tags)
    crf.val(val_words, word_dict, tag_dict, out_path)



    

    
    

