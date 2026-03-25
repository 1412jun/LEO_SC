# from net.decoder import *
# from net.encoder import *
from decoder import *
from encoder import *
from channel import Channel
# from snr_coding import SNRCoding
from snr_coding import snrEncoder
from snr_coding import snrDecoder
from snr_coding import snrChannel
from snr_coding import AdaptiveSNREncoder
from snr_coding import AdaptiveSNRDecoder
from snr_coding import add_noise
from importance_selector import ImportancePredictor
# from denoise2 import UNet2Layer
# from denoise3 import UNet2Layer
# from loss.distortion import Distortion
from distortion import Distortion
# from net.channel import Channel
# from net.channel import Channel
from random import choice
import torch.nn as nn
import torch
global snr

class WITT(nn.Module):
    def __init__(self, args, config):
        super(WITT, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        # self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            global snr
            snr = self.multiple_snr[i]
            snr = float(snr)
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.downsample = config.downsample
        self.model = args.model
        self.use_importance_selector = getattr(args, 'use_importance_selector', False)
        self.importance_alpha = getattr(args, 'importance_alpha', 1.0)
        self.use_adaptive_snr_codec = getattr(args, 'use_adaptive_snr_codec', False)
        self.snr_encoder_path = getattr(args, 'snr_encoder_path', 'snrencoder_-10_model2_1.pth')
        self.snr_decoder_path = getattr(args, 'snr_decoder_path', 'snrdecoder_-10_model2_1.pth')

        if self.use_importance_selector:
            self.importance_selector = ImportancePredictor()

        if self.use_adaptive_snr_codec:
            self.snr_encoder_module = AdaptiveSNREncoder()
            self.snr_decoder_module = AdaptiveSNRDecoder()
        else:
            self.snr_encoder_module = None
            self.snr_decoder_module = None

        self.snr_channel_module = snrChannel().cuda()
        self._legacy_codec_loaded = False

    def _ensure_legacy_codec_loaded(self):
        if self._legacy_codec_loaded:
            return
        self.snr_encoder_module = snrEncoder().cuda()
        self.snr_decoder_module = snrDecoder().cuda()
        self.snr_encoder_module = torch.load(self.snr_encoder_path)
        self.snr_decoder_module = torch.load(self.snr_decoder_path)
        self._legacy_codec_loaded = True

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, given_SNR = None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        feature = self.encoder(input_image, chan_param, self.model)

        if self.use_importance_selector:
            importance_map = self.importance_selector(feature, image=input_image, snr=chan_param)
            feature = feature * (1.0 + self.importance_alpha * importance_map)


        CBR = feature.numel() / 2 / input_image.numel()
        # Feature pass channel

        # if self.pass_channel:
        #     noisy_feature = self.feature_pass_channel(feature, chan_param)
        # else:
        #     noisy_feature = feature
        

        #div2k
        # model = UNet2Layer(in_channels=1, out_channels=1)
        # model_path = '/home/zwz21/下载/WITT/UNet2kl_SNR_20.pth'
        # model = torch.load(model_path)
        # model = model.cuda()
        # recon_image = torch.unsqueeze(noisy_feature, dim=0)
        # recon_image = torch.unsqueeze(noisy_feature, dim=0)
        # output = model(recon_image)
        # noisy_feature = output.squeeze(0)  
        # noisy_feature = output.squeeze(0)

        if not self.use_adaptive_snr_codec:
            self._ensure_legacy_codec_loaded()

        encoded = self.snr_encoder_module(feature, snr)
        transmitted = self.snr_channel_module(encoded, snr)
        noisy_feature = self.snr_decoder_module(transmitted, snr)
    
        # noisy_feature = add_noise(feature, snr)
        # noisy_feature = feature

        # # # # recon_image = self.decoder(feature, chan_param, self.model) #no channel

        recon_image = self.decoder(noisy_feature, chan_param, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()
        
        # return feature, chan_param, self.model, self.decoder
        return feature
 
