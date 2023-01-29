import torch
import torch.nn as nn

#vanila RNN
class RNN(nn.Module):
  def __init__(self,input_size, hidden_size, num_layers,num_classes):
    super(RNN,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_classes = num_classes
    
    #Elman Rnn
    self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, 
                      num_layers = num_layers, batch_first = True, bias = True)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(hidden_size , num_classes)

  def forward(self,x):
    #x.size(0) => batch_Size
    h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    out,_ = self.rnn(x,h_0)
    out = self.fc(out[:][-1][:])
    return out

#LSTM model
class LSTM(nn.Module):
  def __init__(self,input_size, hidden_size, num_layers,num_classes):
    super(RNN,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_classes = num_classes
    
    #Long short term dependency
    self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                      num_layers = num_layers, batch_first = True, bias = True)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(hidden_size , num_classes)

  def forward(self,x):
    #cell state hidden state init
    #x.size(0) => batch_Size
    h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    
    out,(hn, cn) = self.lstm(x,h_0)
    out = self.fc(out[:][-1][:])
    return out