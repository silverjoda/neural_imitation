from policies.policies import *
from dataset import xy_dataset
import yaml
import os
import numpy as np
import torch as T

class DataSetImitator:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/imitate_dataset.yaml"), 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        policy = CVX(obs_dim=23, act_dim=2, hid_dim=64)

        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.config["dataset_path"])
        self.imitate_dataset(dataset_path, policy)

    def imitate_dataset(self, path, policy):
        obs_arr = np.load(path + "/obs.npy")
        act_arr = np.load(path + "/act.npy")
        dataset = xy_dataset.XYDataSet(obs_arr, act_arr)

        loss_fun = T.nn.MSELoss()
        optim = T.optim.Adam(params=policy.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])

        iters = self.config["imitation_iters"]
        for i in range(iters):
            x, y = dataset.get_random_batch(batchsize=self.config["batchsize"])
            y_ = policy(x)
            trn_loss = loss_fun(y_, y)
            if i % 5 == 0:
                with T.no_grad():
                    x_val, y_val = dataset.get_val_data()
                    y_val_ = policy(x_val[:100])
                    val_loss = loss_fun(y_val_, y_val[:100])
                print(f"Iter: {i}/{iters}, trn_loss: {trn_loss},  val_loss:{val_loss}")

            trn_loss.backward()
            optim.step()
            optim.zero_grad()

if __name__ == "__main__":
    dataset_imit = DataSetImitator()