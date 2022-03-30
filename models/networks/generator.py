"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf  # of gen filters in first conv layer

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input  #[1, 184, 256, 256]

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)  #[1, 65536]
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw) #[1, 1024, 8, 8]
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg) #[1, 1024, 8, 8]

        x = self.up(x) #[1, 1024, 16, 16]
        x = self.G_middle_0(x, seg) #[1, 1024, 16, 16]

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg) #[1, 1024, 16, 16]

        x = self.up(x)         #[1, 1024, 32, 32]
        x = self.up_0(x, seg)  #[1, 512, 32, 32]
        x = self.up(x)         #[1, 512, 64, 64]
        x = self.up_1(x, seg)  #[1, 256, 64, 64]
        x = self.up(x)         #[1, 256, 128, 128]
        x = self.up_2(x, seg)  #[1, 128, 128, 128]
        x = self.up(x)         #[1, 128, 256, 256]
        x = self.up_3(x, seg)  #[1, 64, 256, 256]

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1)) #[1, 3, 256, 256]
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []
        
        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            
            
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)



class SPADEPix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=3, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=8, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1) + 4
        # self.input_nc = 4
        norm_layer = get_nonspade_norm_layer(opt,'instance')
        activation = nn.ReLU(False)

        self.initial_conv = [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(self.input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]
        self.initial_conv = nn.Sequential(*self.initial_conv)
        
        # downsample
        mult = 1        
        self.down_0= [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
        self.down_0 = nn.Sequential(*self.down_0)
        mult *= 2
        self.down_1= [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
        self.down_1 = nn.Sequential(*self.down_1)
        mult *= 2
        self.down_2= [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
        self.down_2 = nn.Sequential(*self.down_2)
        mult *= 2
        self.down_3= [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
        self.down_3 = nn.Sequential(*self.down_3)
                       
        # resnet blocks
        self.resnet_blocks=[]
        for i in range(opt.resnet_n_blocks):
            self.resnet_blocks += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]            
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        
        
        self.G_middle_0 = SPADEResnetBlock(mult * opt.ngf, mult * opt.ngf, opt)
         
        # upsample
        self.up=nn.Upsample(scale_factor=2)
        
        nc_in = int(opt.ngf * mult)
        nc_out = int((opt.ngf * mult) / 2)
        self.up_0 = SPADEResnetBlock(nc_in, nc_out, opt)
        
        # mult=mult/2
        nc_in = int(opt.ngf * mult)
        nc_out = int((opt.ngf * mult) / 2)
        self.up_1 = SPADEResnetBlock(nc_in, nc_out, opt)
        
        mult=mult/2
        nc_in = int(opt.ngf * mult)
        nc_out = int((opt.ngf * mult) / 2)
        self.up_2 = SPADEResnetBlock(nc_in, nc_out, opt)         
        
        mult=mult/2
        nc_in = int(opt.ngf * mult)
        nc_out = int((opt.ngf * mult) / 2)
        self.up_3 = SPADEResnetBlock(nc_in, nc_out, opt)       


        # final output conv
        # self.final = [nn.ReflectionPad2d(3),
        #           nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
        #           nn.Tanh()]
        # self.final = nn.Sequential(*self.final)
        
        self.conv_img = nn.Conv2d(nc_out, 3, 3, padding=1)
 
        

    def forward(self, images_masked, mask, seg, z=None):
        input = torch.cat([images_masked, mask, seg], dim=1)

                                     # when opt.ngf=64
        x=self.initial_conv(input)   # [1, 64, 512, 512]
        d0=self.down_0(x)            # [1, 128, 256, 256]        
        d1=self.down_1(d0)           # [1, 256, 128, 128]
        d2=self.down_2(d1)           # [1, 512, 64, 64]
        d3=self.down_3(d2)           # [1, 512, 32, 32]
        
        x=self.resnet_blocks(d3)      # [1, 512, 32, 32], multiple resnet block
        
        x = self.G_middle_0(x, seg)   # [1, 512, 32, 32]
        
        # upsample              
        x=self.up(x)                  # [1, 512, 64, 64]
        x=self.up_0(x,seg)            # [1, 256, 64, 64]
        x=self.up(x)                  # [1, 256, 128, 128]
        x=torch.cat([x, d1], dim=1)   # [1, 512, 128, 128]        
        x=self.up_1(x,seg)           # [1, 256, 128, 128]
        x=self.up(x)                 # [1, 256, 256, 256]
        x=self.up_2(x,seg)           # [1, 128, 256, 256]
        x=self.up(x)                 # [1, 128, 512, 512]
        x=self.up_3(x,seg)           # [1, 64, 512, 512]
                
        x = self.conv_img(F.leaky_relu(x, 2e-1)) #[1, 3, 512, 512]
        x = F.tanh(x)
        
        # x=torch.mul(x,mask)+torch.mul(images_masked,1-mask)
        return x