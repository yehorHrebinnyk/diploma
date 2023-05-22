import torch
import torch.nn as nn


def auto_pad(kernel_size, padding=None):
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
    return padding


class MP(nn.Module):
    def __init__(self, kernel_size=2):
        super(MP, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        return self.layer(x)


class Conv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=1, stride=1, padding=None, groups_count=1,
                 to_activate=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(channel_in,
                              channel_out,
                              kernel_size,
                              stride=stride,
                              padding=auto_pad(kernel_size, padding=padding),
                              groups=groups_count,
                              bias=False)
        self.batch_normalization = nn.BatchNorm2d(channel_out)
        self.activation = nn.SiLU() if to_activate is True else (
            to_activate if isinstance(to_activate, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normalization(x)
        return self.activation(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class SPPCSPC(nn.Module):
    def __init__(self, channel_in, channel_out, n=1, shortcut=False, groups_count=1, e=0.5, kernels=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        hidden_channels = int(2 * channel_out * e)
        self.cv1 = Conv(channel_in, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = Conv(channel_in, hidden_channels, kernel_size=1, stride=1)
        self.cv3 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1)
        self.cv4 = Conv(hidden_channels, hidden_channels, kernel_size=1, stride=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2) for kernel in kernels])
        self.cv5 = Conv(4 * hidden_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv6 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1)
        self.cv7 = Conv(2 * hidden_channels, channel_out, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


class IDetect(nn.Module):
    stride = None
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, number_classes=80, anchors=(), ch=()):
        super(IDetect, self).__init__()
        self.number_classes = number_classes
        self.number_outputs = self.number_classes + 5  # number_classes + x, y, w, h, obj
        self.number_layers = len(anchors)  # number of outputs
        self.number_anchors = len(anchors[0]) // 2  # dividing by 2 because of x, y coordinates in one list
        self.grid = [torch.zeros(1)] * self.number_layers
        a = torch.tensor(anchors).float().view(self.number_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchors_grid', a.clone().view(self.number_layers, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.conv_outputs = nn.ModuleList(nn.Conv2d(x, self.number_outputs * self.number_anchors, 1) for x in ch)
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.number_outputs * self.number_anchors) for _ in ch)

    def forward(self, x):
        z = []

        for i in range(self.number_layers):
            x[i] = self.conv_outputs[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.number_anchors, self.number_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchors_grid[i]
                z.append(y.view(bs, -1, self.number_outputs))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
