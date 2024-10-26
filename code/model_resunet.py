import torch
import torch.nn as nn
import torch.nn.functional as F

#Double conv
class Double_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, res = True):
        super(Double_Conv2d, self).__init__()
        self.res = res
        self.leakyrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Dropout(0.5)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5)
            )
        if self.res:
            if in_channel != out_channel:
                self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding='same', bias=False)

    def forward(self, inp):
        res = inp
        out = self.conv1(inp)
        out = self.conv2(out)
        if self.res:
            if hasattr(self, "conv3"):
                res = self.conv3(res)
            out += res
            
        out = self.leakyrelu(out)
        return out
    
# UpBlock
class Up_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up_Block, self).__init__()
        self.conv = Double_Conv2d(in_channel+out_channel, out_channel, res = True)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs2, inputs1):
        results2 = self.up(inputs2)  # upsampling
        padding = (results2.size()[-1] - inputs1.size()[-1]) // 2  # shape(batch, channel, width, height)
        results1 = F.pad(inputs1, 2 * [padding, padding])
        results = torch.cat([results1, results2], 1) 
        return self.conv(results)

class ResUNet(nn.Module):
    def __init__(self, in_channel = 6, base_filters = 64, ndepth = 5):
        super(ResUNet, self).__init__()
        self.in_channel   = in_channel
        self.base_filters = base_filters
        self.ndepth       = ndepth
        
        self.MSELoss      = nn.MSELoss()
        self.rain_vec     = nn.Linear(9, 4096)
        self.leakyrelu    = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        self.convf        = nn.Conv2d(self.base_filters, 1, kernel_size=1, padding='same')
        
        #filters = [64, 128, 256, 512, 512]
        filters = []
        for i in range(self.ndepth):
            a = self.base_filters * (2**i)
            if a > 512:
                a = 512
            filters.append(a)
        
        self.conv0 = Double_Conv2d(self.in_channel, filters[0], res = True)
        for i in range(1, self.ndepth):
            setattr(self, "conv%d"%(i), Double_Conv2d(filters[i-1], filters[i], res = True))
        
        for i in range(self.ndepth):
            setattr(self, "avgpool%d"%(i), nn.AvgPool2d(kernel_size=2))
        
        self.raindim   = int(256/(2**self.ndepth))
        self.raindepth = int(4096/(self.raindim**2))
        
        setattr(self, "upsample%d"%(self.ndepth-1), Up_Block(filters[self.ndepth-1]+self.raindepth, filters[self.ndepth-1]))
        for i in range(self.ndepth-2, -1, -1):
            setattr(self, "upsample%d"%(i), Up_Block(filters[i+1], filters[i]))
        
        print("---model config---")
        print(" filters ",filters)
        print(" raindim ", self.raindim)
        print(" raindepth ", self.raindepth)

    def forward(self, inputs, rain_p, labels = None):
        
        batch_size = inputs.shape[0]
        skip_x = []
        for i in range(self.ndepth):
            if i == 0:
                x = self.conv0(inputs)
            else:
                x = getattr(self, "conv%d"%(i))(a)
            skip_x.append(x)
            a = getattr(self, "avgpool%d"%(i))(x)
        
        rain_vector = self.rain_vec(rain_p)
        rain_vector = self.leakyrelu(rain_vector)
        rain_vector = torch.reshape(rain_vector, (batch_size, self.raindepth, self.raindim, self.raindim))

        d = torch.cat([a, rain_vector], dim = 1)
        for i in range(self.ndepth-1, -1, -1):
            d = getattr(self, "upsample%d"%(i))(d, skip_x[i])
        
        results = self.convf(d)
        results = torch.squeeze(results, 1)
        
        if labels is not None:
            labels = labels.view(results.shape)
            mseloss  = self.MSELoss(results, labels)
        else:
            mseloss = None
        output = {
            "loss": mseloss,
            "results": results
        }
        return output