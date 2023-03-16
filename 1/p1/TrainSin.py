r""" This file is to train and eval the sin-model """

import math
import random
import matplotlib.pyplot as plt
import numpy as np
from RegressModel import RegressionNet


def eval(model, eval_data_size=500, if_draw=False):
    model.eval_data = []
    for i in range(0, eval_data_size):
        model.eval_data.append(random.uniform(-math.pi, math.pi))
    model.eval_data.sort()

    total_loss = 0
    eval_results = []
    for i in range(0, eval_data_size):
        pred = model.forward(np.array([[model.eval_data[i]]]))
        eval_results.append(pred[0][0])
        total_loss += abs(np.sin(model.eval_data[i])- pred[0][0])
    avg_loss = total_loss / eval_data_size
    print("eval accuracy, %f total loss in %d data size" 
          % (total_loss, eval_data_size))
    print("Avg_loss : ", avg_loss, '\n')

    if if_draw:
        plt.plot(model.eval_data, np.sin(model.eval_data),
                 color="red", label="Real Sin")
        plt.plot(model.eval_data, eval_results, 
                 color="green", label="My Regression")
        plt.xlim(-math.pi, math.pi)
        plt.ylim(-1, 1)
        plt.title("With Loss: " + str(avg_loss))
        plt.legend()
        plt.show()

    return avg_loss


if __name__ == "__main__":
    BPsin = RegressionNet(layer_arch=[1,64,64,1], lr=0.01, batch_size=20)

    epoch_record_x = []
    avg_loss_record_y = []
    epoch_record_x.append(0)
    avg_loss_record_y.append(eval(BPsin, eval_data_size=500, if_draw=True))

    for epoch in range(0,10001):

        np.random.shuffle(BPsin.train_data)
        gt_result = np.sin(BPsin.train_data)
        batch_num = 0
        for i in range(0, len(BPsin.train_data)):
            pred = BPsin.forward(np.array([[BPsin.train_data[i]]]))
            loss = pred - np.array([[gt_result[i]]])
            BPsin.backward(loss)
            batch_num += 1
            if batch_num == BPsin.batch_size:
                batch_num = 0
                BPsin.update_weight(BPsin.lr) 
                # it seems that we can make lr/=2
        
        if epoch % 200 == 0:
            print("Epoch" , epoch)
            if_draw = False
            # if epoch % 1000 == 0:
            #     if_draw = True
            if epoch == 10000:
                if_draw = True
            epoch_record_x.append(epoch)
            avg_loss_record_y.append(eval(BPsin, eval_data_size=500, if_draw=if_draw))
    
    plt.plot(epoch_record_x, avg_loss_record_y)
    plt.title("Loss with epoch")
    plt.show()