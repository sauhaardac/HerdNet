from tensorboardX import SummaryWriter

class Logger:
    def __init__(self):
        run_name = input('Run/Test Name (Descriptive): ')
        self.writer = SummaryWriter(f"logs/{run_name}")

    def episode_score(self, score, ep):
        # add 20 to make the score always positive!
        self.writer.add_scalar('data/epscore', score + 20, ep)
