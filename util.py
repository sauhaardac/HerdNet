from tensorboardX import SummaryWriter
import os
import torch

class Logger:
    def __init__(self):
        self.run_name = input('Run/Test Name (Descriptive): ')
        self.writer = SummaryWriter(f"logs/{self.run_name}")
        if not os.path.isdir(f"saves/{self.run_name}"):
            os.mkdir(f"saves/{self.run_name}")

    def episode_score(self, score, ep):
        # add 20 to make the score always positive!
        self.writer.add_scalar('data/epscore', score + 20, ep)

    def save_model(self, score, weights, ep):
        torch.save(weights, f"saves/{self.run_name}/{score:2.2f}-{ep}.save")
