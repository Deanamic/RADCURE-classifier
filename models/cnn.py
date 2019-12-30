import torch
import torch.nn as nn

'''
CNN using Conv3d and 3 fully connected layers
'''
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.n_convLayers = config['conv_layers']
        self.skip = config['skip_layers']
        self.n_linearLayers = config['linear_layers']
        self.image_size = config['input_scale_size']
        dropout_rate = config['dropout_rate']
        slope = config['leakyrelu_param']
        self.convLayers = nn.ModuleList()
        curIn = 1
        curOut = 2
        for i in range(self.n_convLayers):
            inC = curIn
            curOut = curOut * 2
            outC = curOut-1 if self.skip else curOut
            curIn = curOut
            self.convLayers.append(self.Convolutional_Layer(inC, outC, 3, 1, dropout_rate, slope, 'M' if i + 1 < self.n_convLayers else 'A'))
            self.image_size = self.image_size // 2

        self.linearLayers = nn.ModuleList()
        inC = curOut * self.image_size * self.image_size * self.image_size
        outC = curOut // 2
        for i in range(self.n_linearLayers-1):
            curLayer = nn.Sequential(
                nn.Linear(inC, outC),
                nn.LeakyReLU(slope)
            )
            self.linearLayers.append(curLayer)
            curOut = outC
            inC = curOut
            outC = curOut // 2

        self.outLayer = nn.Sequential(
            nn.Linear(curOut, 1),
            nn.Sigmoid()
        )
        self.pooling = nn.AvgPool3d(2);

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        y = x
        for i in range(self.n_convLayers):
            y = self.pooling(y)
            x = self.convLayers[i](x)
            if(self.skip):
                x = torch.cat([x,y], dim = 1)

        x = x.view(x.size(0), -1)
        for i in range(self.n_linearLayers-1):
            x = self.linearLayers[i](x)

        x = self.outLayer(x)
        return x

    def Convolutional_Layer(self, inputC, outputC, kernel_size, padding, dropout_rate, slope, pooling):
        return nn.Sequential(
            nn.Conv3d(inputC, outputC, kernel_size = kernel_size, padding = padding),
            nn.Dropout3d(dropout_rate),
            nn.LeakyReLU(slope),
            nn.BatchNorm3d(outputC),
            nn.MaxPool3d(2) if pooling == 'M' else nn.AvgPool3d(2)
        )
