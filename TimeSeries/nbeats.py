import torch
import torch.nn as nn
import torch.optim as optim

class Block(nn.Module):
    def __init__(self, config,loopback,blockType):
        super(Block,self).__init__()
        self.inputSize = config.window_size
        self.predictSize = config.horizon
        self.theta = 32
        self.blockType = blockType
        self.loopback = loopback * 6

        self.fcStack = nn.Sequential(
            nn.Linear(self.loopback, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.genericBackCastFC = nn.Sequential(
            nn.Linear(512, self.theta),
            nn.ReLU(),
            nn.Linear(self.theta, self.loopback)
        )
        self.genericForeCastFC = nn.Sequential(
            nn.Linear(512, self.theta),
            nn.ReLU(),
            nn.Linear(self.theta, self.predictSize)
        )

    def forward(self,x):# x.shape = torch.Size ( batchSize , observation)
        out = self.fcStack(x) # ( batchSize * 256 )

        backcast = self.genericBackCastFC(out)
        forecast = self.genericForeCastFC(out)
        return backcast, forecast

class Stack(nn.Module):
    def __init__(self, config,loopback, blockType, device):
        super(Stack,self).__init__()
        self.device = device
        self.config = config
        self.loopback = loopback
        #  depth 5 stack
        self.block = nn.ModuleList([Block(config,loopback,'generic').to(device) for i in range(30)])
        

    def forward(self,x):
        back = torch.zeros(x.size(0), x.size(1)).to(self.device)
        fore = torch.zeros(x.size(0), self.config.horizon).to(self.device)
        back = x

        for i in range(30):
            outBack, outFore = self.block[i](back)
            back = back - outBack
            fore = outFore + fore

        return back, fore

