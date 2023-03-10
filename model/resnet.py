import torch
import torch.nn as nn
# ResNet
"""
  skip-connection :
    y = f(x) + x
  
  1x1 convolution의 역할 :
    channel의 축소
    ex ) 28 * 28 * 192  conv  1 * 1 * 16
      => 28 * 28 * 16으로 데이터 줄이기 가능

"""
class ResidualNet(nn.Module):
  def __init__(self, output_dim):
    super(ResidualNet, self).__init__()

    self.n_classes = output_dim
    # 16, 1, 28, 28
    self.conv1 = nn.Conv2d(1,8,kernel_size = 3, stride =1, padding = 1)
    # 16, 128, 28, 28
    self.block = nn.Sequential(
        nn.Conv2d(8,4,kernel_size = 1,stride = 1),
        nn.ReLU(),
        nn.Conv2d(4,4, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(4,8, kernel_size = 1, stride = 1),
    )

    self.softmax = nn.Softmax(dim = -1)
    self.fc = nn.Linear(8* 28 * 28,self.n_classes)

  def forward(self,x):
    x = self.conv1(x)
    identity = x
    out = self.block(x)
    out += identity
    #flatten
    out = out.view(x.size(0),-1)
    out = self.softmax(out)
    out = self.fc(out)
    return out