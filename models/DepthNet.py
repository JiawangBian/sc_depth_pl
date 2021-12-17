import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_encoder import ResnetEncoder


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]

        # decoder
        self.upconvs0 = []
        self.upconvs1 = []
        self.dispconvs = []
        self.i_to_scaleIdx_conversion = {}

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs0.append(ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs1.append(ConvBlock(num_ch_in, num_ch_out))

        for cnt, s in enumerate(self.scales):
            self.dispconvs.append(
                Conv3x3(self.num_ch_dec[s], self.num_output_channels))

            if s in range(4, -1, -1):
                self.i_to_scaleIdx_conversion[s] = cnt

        self.upconvs0 = nn.ModuleList(self.upconvs0)
        self.upconvs1 = nn.ModuleList(self.upconvs1)
        self.dispconvs = nn.ModuleList(self.dispconvs)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        return

    def forward(self, input_features):

        self.outputs = []

        # decoder
        x = input_features[-1]

        for cnt, i in enumerate(range(4, -1, -1)):
            x = self.upconvs0[cnt](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.upconvs1[cnt](x)
            if i in self.scales:
                idx = self.i_to_scaleIdx_conversion[i]
                disp = self.alpha * \
                    self.sigmoid(self.dispconvs[idx](x)) + self.beta
                depth = 1.0 / disp
                self.outputs.append(depth)

        self.outputs = self.outputs[::-1]
        return self.outputs


class DepthNet(nn.Module):

    def __init__(self, num_layers=18, pretrained=True):
        super(DepthNet, self).__init__()
        self.encoder = ResnetEncoder(
            num_layers=num_layers, pretrained=pretrained, num_input_images=1)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)

    def init_weights(self):
        pass

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs[0]


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = DepthNet().cuda()
    model.train()

    B = 4

    tgt_img = torch.randn(B, 3, 256, 832).cuda()

    tgt_depth = model(tgt_img)

    print(tgt_depth.size())
