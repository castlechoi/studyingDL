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

class NBEATS(nn.Module):
    def __init__(self, config, device):
        super(NBEATS, self).__init__()
        #  depth 5 stack
        self.device = device
        self.config = config
        self.stack2 = Stack(config,2,'generic',device).to(device)
        self.stack3 = Stack(config,3,'generic',device).to(device)
        self.stack4 = Stack(config,4,'generic',device).to(device)
        self.stack5 = Stack(config,5,'generic',device).to(device)
        self.stack6 = Stack(config,6,'generic',device).to(device)
        self.stack7 = Stack(config,7,'generic',device).to(device)
        

    def forward(self,x):
        x = x.view(x.size(0), -1)
        _, fore2 = self.stack2(x[:,30:]) 
        
        _, fore3 = self.stack3(x[:,24:])
        
        _, fore4 = self.stack4(x[:,18:])
        
        _, fore5 = self.stack5(x[:,12:])
        
        _, fore6 = self.stack6(x[:,6:])
        
        _, fore7 = self.stack7(x)
        return (fore2 + fore3 + fore4 + fore5 + fore6 + fore7) / 6