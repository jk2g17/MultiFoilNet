import torch
import torch.nn as nn


# Initialise Weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def blockUNet(in_c, out_c, name, transposed=False, bn=True, AFunc="ReLu", size=4, pad=1, dropout=0.):
    
    # Create Sequential Block.
    block = nn.Sequential()
    
    # Specify the required activation function.
    if AFunc == "ReLu":
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    elif AFunc == "LReLu":
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    elif AFunc == "Sigmoid":
        block.add_module('%s_sigmoid' % name, nn.Sigmoid())
    elif AFunc == "TanH":
        block.add_module('%s_tanh' % name, nn.Tanh())
        
    # If the block is on the downsampling path, use a standard Conv2d function. If upsampling is required, first initialise the Upsample function and then apply Conv2d.
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear'))
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
        
    # If batch normalisation is reqired, apply it.
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
        
    # If dropout (noise) is required, initialise the Dropout2d function.
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block
    
# MultiFoilNet Model.
class MultiFoilNet(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(MultiFoilNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        # First block.
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(1, channels, 4, 2, 1, bias=True))

        # Encoder section.
        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  AFunc="LReLu", dropout=dropout )
        self.layer3= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  AFunc="LReLu", dropout=dropout )
        self.layer4 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  AFunc="LReLu", dropout=dropout )
        self.layer5 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  AFunc="LReLu", dropout=dropout )
        self.layer6 = blockUNet(channels*8, channels*16, 'layer5', transposed=False, bn=True,  AFunc="LReLu", dropout=dropout , size=2,pad=0)
        self.layer7 = blockUNet(channels*16, channels*16, 'layer6', transposed=False, bn=False, AFunc="LReLu", dropout=dropout , size=2,pad=0)
        self.layer8 = blockUNet(channels*16, channels*16, 'layer6', transposed=False, bn=False, AFunc="LReLu", dropout=dropout , size=2,pad=0)
        self.layer9 = blockUNet(channels*16, channels*16, 'layer6', transposed=False, bn=False, AFunc="LReLu", dropout=dropout , size=2,pad=0)
        
        # Decoder section.
        self.dlayer9 = blockUNet(channels*16, channels*16, 'dlayer6', transposed=True, bn=True, AFunc="ReLu", dropout=dropout , size=2,pad=0)        
        self.dlayer8 = blockUNet(channels*32, channels*16, 'dlayer6', transposed=True, bn=True, AFunc="ReLu", dropout=dropout , size=2,pad=0)        
        self.dlayer7 = blockUNet(channels*32, channels*16, 'dlayer6', transposed=True, bn=True, AFunc="ReLu", dropout=dropout , size=2,pad=0)
        self.dlayer6 = blockUNet(channels*32,channels*8, 'dlayer5', transposed=True, bn=True, AFunc="ReLu", dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, AFunc="ReLu", dropout=dropout ) 
        self.dlayer4 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, AFunc="ReLu", dropout=dropout )
        self.dlayer3= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, AFunc="ReLu", dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, AFunc="ReLu", dropout=dropout )

        # Final block.
        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

    # Define the standard normal pass through MultiFoilNet.
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)

        dout9 = self.dlayer9(out9)
        dout9_out8 = torch.cat([dout9, out8], 1)
        dout8 = self.dlayer8(dout9_out8)
        dout8_out7 = torch.cat([dout8, out7], 1)        
        dout7 = self.dlayer7(dout8_out7)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4= self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

# MultiFoilNet Model with TanH Activation Function.
class MultiFoilNet_TanH(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(MultiFoilNet_TanH, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        # First block.
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(1, channels, 4, 2, 1, bias=True))

        # Encoder section.
        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  AFunc="TanH", dropout=dropout )
        self.layer3= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  AFunc="TanH", dropout=dropout )
        self.layer4 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  AFunc="TanH", dropout=dropout )
        self.layer5 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  AFunc="TanH", dropout=dropout )
        self.layer6 = blockUNet(channels*8, channels*16, 'layer5', transposed=False, bn=True,  AFunc="TanH", dropout=dropout , size=2,pad=0)
        self.layer7 = blockUNet(channels*16, channels*16, 'layer6', transposed=False, bn=False, AFunc="TanH", dropout=dropout , size=2,pad=0)
        self.layer8 = blockUNet(channels*16, channels*16, 'layer6', transposed=False, bn=False, AFunc="TanH", dropout=dropout , size=2,pad=0)
        self.layer9 = blockUNet(channels*16, channels*16, 'layer6', transposed=False, bn=False, AFunc="TanH", dropout=dropout , size=2,pad=0)
        
        # Decoder section.
        self.dlayer9 = blockUNet(channels*16, channels*16, 'dlayer6', transposed=True, bn=True, AFunc="TanH", dropout=dropout , size=2,pad=0)        
        self.dlayer8 = blockUNet(channels*32, channels*16, 'dlayer6', transposed=True, bn=True, AFunc="TanH", dropout=dropout , size=2,pad=0)        
        self.dlayer7 = blockUNet(channels*32, channels*16, 'dlayer6', transposed=True, bn=True, AFunc="TanH", dropout=dropout , size=2,pad=0)
        self.dlayer6 = blockUNet(channels*32,channels*8, 'dlayer5', transposed=True, bn=True, AFunc="TanH", dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, AFunc="TanH", dropout=dropout ) 
        self.dlayer4 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, AFunc="TanH", dropout=dropout )
        self.dlayer3= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, AFunc="TanH", dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, AFunc="TanH", dropout=dropout )

        # Final block.
        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.Tanh())
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

    # Define the standard normal pass through MultiFoilNet.
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)

        dout9 = self.dlayer9(out9)
        dout9_out8 = torch.cat([dout9, out8], 1)
        dout8 = self.dlayer8(dout9_out8)
        dout8_out7 = torch.cat([dout8, out7], 1)        
        dout7 = self.dlayer7(dout8_out7)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4= self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1