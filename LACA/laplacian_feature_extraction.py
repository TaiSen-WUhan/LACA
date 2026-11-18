import torch
import torch.nn as nn
import torch.nn.functional as F

def disp2disp_gradient_xy(disp):
    # 计算水平垂直梯度
    N, _, H, W = disp.shape
    disp_pad = F.pad(disp, pad=(1, 1, 1, 1), mode='replicate')

    # grad kernel
    kernel = torch.zeros((2, 1, 3, 3))
    kernel[:, :, 1, 1] = -1
    vus = [(1, 2), (2, 1)]
    # 相当于[-1,1]与[[1],[-1]]
    for i, vu in enumerate(vus):
        v, u = vu
        kernel[i, :, v, u] = 1

    grads = F.conv2d(disp_pad.repeat(1, 2, 1, 1), kernel.to(disp.device), padding=0, groups=2)  # n,2,h,w
    return grads

class LaplacianFeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(2, 32, kernel_size=5, padding=2, stride=1, bias=True),
                                   nn.InstanceNorm2d(32), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=True),
                                   nn.InstanceNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=True),
                                   nn.InstanceNorm2d(64), nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=True),
                                   nn.InstanceNorm2d(128), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=True),
                                   nn.InstanceNorm2d(128), nn.ReLU())
    def forward(self, depth):
        grads = disp2disp_gradient_xy(depth)
        grads_xx = disp2disp_gradient_xy(grads[:, :1])[:, :1]
        grads_yy = disp2disp_gradient_xy(grads[:, 1:])[:, 1:]

        x = torch.cat((grads_xx, grads_yy), dim = 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + self.conv5(x)

        out = [x, F.avg_pool2d(x, kernel_size=2, stride=2)]
        return out