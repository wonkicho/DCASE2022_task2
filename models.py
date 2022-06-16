from torch import nn
import timm
import torch
import torch.nn.functional as F
import math
import librosa
import os
from torch.nn import Parameter
from torchsummary import summary
from utils import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

Mobilenetv2_bottleneck_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50,easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size()).to(x.device)
        #print(x.device, label.device, one_hot.device)
        #one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = label.long()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)
    
class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 arcface=None):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(2, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)

        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        #(8, 27)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)
        
        self.fc_out = nn.Linear(128, num_class)
        self.arcface = arcface
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, label):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        if self.arcface is not None:
            out = self.arcface(feature, label)
        else:
            out = self.fc_out(feature)
        return out, feature

class CustomClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False, arcface=None):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        self.model.conv_stem = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.in_features = self.model.classifier.in_features
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(0.2, inplace=True)
        self.fc1 = nn.Linear(self.in_features , 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc_out = nn.Linear(128, n_class) 
        self.arcface = arcface
        

        # if self.arcface:
        #     self.model.classifier = self.arcface(n_features, n_class)
        # else:
        #     self.model.classifier = nn.Linear(n_features, n_class)
        
    def forward(self, x, label):
        features = self.model.forward_features(x)
        features = self.bn1(features)
        features = self.dropout(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        features = self.fc2(features)
        features = self.bn3(features)
        #features = F.normalize(features)
        if self.arcface:
            return self.arcface(features, label), features
        else:
            out = self.fc_out(features)
            return out, features
        
    
class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(313),#313
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False)
            ) for _ in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out  




class STgramMFN(nn.Module):
    def __init__(self, num_class,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 arcface=None):
        super(STgramMFN, self).__init__()
        self.arcface = arcface
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_class,
                                           bottleneck_setting=bottleneck_setting,
                                           arcface=arcface)
        # self.custom_clf = CustomClassifier(model_arch = "mobilenetv3_small_100",
        #                                   n_class = num_class, pretrained = True,
        #                                   arcface = arcface
        #                                 )

        
    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label):
        x_wav = self.tgramnet(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_wav), dim=1)
        out, feature = self.mobilefacenet(x, label)
        #out, feature = self.custom_clf(x, label)
        return out, feature
    

if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    yaml_path = "./config.yaml"
    CONFIG = yaml_load(yaml_path)
    arcface = ArcMarginProduct(128, 6, m=CONFIG["arcface"]["m"], s=CONFIG["arcface"]["s"])
    model = STgramMFN(num_class=6,
                    c_dim=CONFIG["feature"]["n_mels"],
                    win_len=CONFIG["feature"]["win_length"],
                    hop_len=CONFIG["feature"]["hop_length"],
                    arcface=arcface).to(DEVICE)
    
    summary(model, [(1, 160000),(1, 128, 313),(0, 6)])
    