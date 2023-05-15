import torch
import torch.nn as nn
import torch.optim as optim

class Block(nn.Module):
    def __init__(self, config,blockType):
        super(Block,self).__init__()
        self.inputSize = config.window_size
        self.predictSize = config.horizon
        self.theta = config.theta
        self.blockType = blockType

        self.fcStack = nn.Sequential(
            nn.Linear(self.inputSize, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.genericBackCastFC = nn.Sequential(
            nn.Linear(256, self.theta),
            nn.Linear(self.theta, self.inputSize)
        )
        self.genericForeCastFC = nn.Sequential(
            nn.Linear(256, self.theta),
            nn.Linear(self.theta, self.predictSize)
        )

    def forward(self,x):  # x.shape = torch.Size ( batchSize , observation)
        out = self.fcStack(x) # ( batchSize * 256 )

        backcast = self.genericBackCastFC(out)
        forecast = self.genericForeCastFC(out)
        return backcast, forecast

class Stack(nn.Module):
    def __init__(self, config, blockType, device):
        super(Stack,self).__init__()
        self.device = device
        self.config = config
        #  depth 5 stack
        self.block1 = Block(config,'generic').to(device)
        self.block2 = Block(config,'generic').to(device)
        self.block3 = Block(config,'generic').to(device)
        self.block4 = Block(config,'generic').to(device)
        self.block5 = Block(config,'generic').to(device)

    def forward(self,x):
        back = torch.zeros(x.size(0),x.size(1)).to(self.device)
        fore = torch.zeros(x.size(0), self.config.horizon).to(self.device)

        outBack, outFore = self.block1(x)
        back = x - outBack
        fore += outFore

        outBack, outFore = self.block2(x)
        back -= outBack
        fore += outFore

        outBack, outFore = self.block3(x)
        back -= outBack
        fore += outFore

        outBack, outFore = self.block4(x)
        back -= outBack
        fore += outFore

        outBack, outFore = self.block5(x)
        back -= outBack
        fore += outFore
        return back, fore

class NBEATS(nn.Module):
    def __init__(self, config, device):
        super(NBEATS, self).__init__()
        #  depth 5 stack
        self.device = device
        self.config = config
        self.stack1 = Stack(config,'generic',device).to(device)


    def forward(self,x):
        x = x.view(x.size(0), -1)
        back = torch.zeros(x.size(0),x.size(1)).to(self.device)
        fore = torch.zeros(x.size(0),self.config.horizon).to(self.device)

        outBack, outFore = self.stack1(x)

        back = x - outBack
        fore += outFore
        return fore