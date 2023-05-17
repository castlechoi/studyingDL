import torch
import torch.nn as nn
class sMAPE(nn.Module):
    def __init__(self, horizon):
        super(sMAPE, self).__init__()
        self.horizon = horizon
  
    def forward(self, src,tgt):
        tot = 0
        s = src.view(-1,self.horizon)
        t = tgt.view(-1,self.horizon)
        
        up = torch.abs(s - t)
        down =torch.abs(s) + torch.abs(t)
        tot = torch.mean( up / down, -1) * 200
        return  torch.mean(tot)
    
class MAPE(nn.Module):
    def __init__(self, horizon):
        super(MAPE, self).__init__()
        self.horizon = horizon
        
    def forward(self, src, tgt):
        tot = 0
        s = src.view(-1,self.horizon) # src ( batchSize, Horizon )
        t = tgt.view(-1,self.horizon) # tgt ( batchSize, Horizon )
        
        up = torch.abs(s - t)
        down = torch.abs(t)
        tot = torch.mean( up / down, -1) * 100
        return torch.mean(tot)
    
class MASE(nn.Module):
    def __init__(self, horizon, window_size,m):
        super(MASE, self).__init__()
        self.horizon = horizon
        self.window_size = window_size
        self.m = m # yearly for 1
        
    def forward(self, src, tgt, obv): # pred, labels
        tot = 0
        s = src.view(-1,self.horizon) # src ( batchSize, Horizon )
        t = tgt.view(-1,self.horizon) # tgt ( batchSize, Horizon )
        o = obv.view(-1,self.window_size) # ( batchSize, Observation)
        
        o = torch.cat((o,t), dim = 1)
        
        up = torch.abs(s - t)
        down = torch.abs(o[:,1] - o[:,0])
        for i in range(2,o.size(1)):
            down += torch.abs((o[:,i] - o[:,i-1] ))
        down = down.view(-1,1)
        
        tot = up/down
        tot = torch.mean(tot, dim = 1) # loss for each batch
        return torch.mean(tot) # loss for total batch
        
        
    
        