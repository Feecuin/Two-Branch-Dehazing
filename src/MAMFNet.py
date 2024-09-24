import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_value = self.avg_pool(x).view(x.size(0), -1)
        max_value = self.max_pool(x).view(x.size(0), -1)
        channel_attention = self.fc(avg_value + max_value).view(x.size(0), x.size(1), 1, 1)

        return channel_attention 

class PixelAttention(nn.Module):
    def __init__(self):
        super(PixelAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool, max_pool], dim=1)

        pixel_attention = self.conv1(combined)
        pixel_attention = self.sigmoid(pixel_attention)

        return pixel_attention 

class ChannelPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelPixelAttention, self).__init__()

        self.channel_attention = ChannelAttention(in_channels)
        self.pixel_attention = PixelAttention()

    def forward(self, x):
        channel_att = self.channel_attention(x)
        pixel_att = self.pixel_attention(x)
        return channel_att * pixel_att * x




class RLN(nn.Module):
	r"""Revised LayerNorm"""
	def __init__(self, dim, eps=1e-5, detach_grad=False):
		super(RLN, self).__init__()
		self.eps = eps
		self.detach_grad = detach_grad

		self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
		self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

		self.meta1 = nn.Conv2d(1, dim, 1)
		self.meta2 = nn.Conv2d(1, dim, 1)

		trunc_normal_(self.meta1.weight, std=.02)
		nn.init.constant_(self.meta1.bias, 1)

		trunc_normal_(self.meta2.weight, std=.02)
		nn.init.constant_(self.meta2.bias, 0)

	def forward(self, input):
		mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
		std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

		normalized_input = (input - mean) / std

		if self.detach_grad:
			rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
		else:
			rescale, rebias = self.meta1(std), self.meta2(mean)

		out = normalized_input * self.weight + self.bias
		return out, rescale, rebias


class Mlp(nn.Module):
	def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.network_depth = network_depth

		self.mlp = nn.Sequential(
			nn.Conv2d(in_features, hidden_features, 1),
			nn.ReLU(True),
			nn.Conv2d(hidden_features, out_features, 1)
		)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.network_depth) ** (-1/4)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.mlp(x)


def window_partition(x, window_size):
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
	return windows


def window_reverse(windows, window_size, H, W):
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x


def get_relative_positions(window_size):
	coords_h = torch.arange(window_size)
	coords_w = torch.arange(window_size)

	coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
	coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
	relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

	relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
	relative_positions_log  = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

	return relative_positions_log


class WindowAttention(nn.Module):
	def __init__(self, dim, window_size, num_heads):

		super().__init__()
		self.dim = dim
		self.window_size = window_size  # Wh, Ww
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		relative_positions = get_relative_positions(self.window_size)
		self.register_buffer("relative_positions", relative_positions)
		self.meta = nn.Sequential(
			nn.Linear(2, 256, bias=True),
			nn.ReLU(True),
			nn.Linear(256, num_heads, bias=True)
		)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, qkv):
		B_, N, _ = qkv.shape

		qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

		q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))

		relative_position_bias = self.meta(self.relative_positions)
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
		attn = attn + relative_position_bias.unsqueeze(0)

		attn = self.softmax(attn)

		x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
		return x


class Attention(nn.Module):
	def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
		super().__init__()
		self.dim = dim
		self.head_dim = int(dim // num_heads)
		self.num_heads = num_heads

		self.window_size = window_size
		self.shift_size = shift_size

		self.network_depth = network_depth
		self.use_attn = use_attn
		self.conv_type = conv_type

		if self.conv_type == 'Conv':
			self.conv = nn.Sequential(
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
				nn.ReLU(True),
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
			)

		if self.conv_type == 'DWConv':
			self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

		if self.conv_type == 'DWConv' or self.use_attn:
			self.V = nn.Conv2d(dim, dim, 1)
			self.proj = nn.Conv2d(dim, dim, 1)

		if self.use_attn:
			self.QK = nn.Conv2d(dim, dim * 2, 1)
			self.attn = WindowAttention(dim, window_size, num_heads)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			w_shape = m.weight.shape
			
			if w_shape[0] == self.dim * 2:	# QK
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)		
			else:
				gain = (8 * self.network_depth) ** (-1/4)
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def check_size(self, x, shift=False):
		_, _, h, w = x.size()
		mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
		mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

		if shift:
			x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
						  self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
		else:
			x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x

	def forward(self, X):
		B, C, H, W = X.shape

		if self.conv_type == 'DWConv' or self.use_attn:
			V = self.V(X)

		if self.use_attn:
			QK = self.QK(X)
			QKV = torch.cat([QK, V], dim=1)

			# shift
			shifted_QKV = self.check_size(QKV, self.shift_size > 0)
			Ht, Wt = shifted_QKV.shape[2:]

			# partition windows
			shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
			qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

			attn_windows = self.attn(qkv)

			# merge windows
			shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

			# reverse cyclic shift
			out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
			attn_out = out.permute(0, 3, 1, 2)

			if self.conv_type in ['Conv', 'DWConv']:
				conv_out = self.conv(V)
				out = self.proj(conv_out + attn_out)
			else:
				out = self.proj(attn_out)

		else:
			if self.conv_type == 'Conv':
				out = self.conv(X)				# no attention and use conv, no projection
			elif self.conv_type == 'DWConv':
				out = self.proj(self.conv(V))

		return out


class TransformerBlock(nn.Module):
	def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
				 norm_layer=nn.LayerNorm, mlp_norm=False,
				 window_size=8, shift_size=0, use_attn=True, conv_type=None):
		super().__init__()
		self.use_attn = use_attn
		self.mlp_norm = mlp_norm

		self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
		self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
							  shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

		self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
		self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

	def forward(self, x):
		identity = x
		if self.use_attn: x, rescale, rebias = self.norm1(x)
		x = self.attn(x)
		if self.use_attn: x = x * rescale + rebias
		x = identity + x

		identity = x
		if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
		x = self.mlp(x)
		if self.use_attn and self.mlp_norm: x = x * rescale + rebias
		x = identity + x
		return x


class BasicLayer(nn.Module):
	def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
				 norm_layer=nn.LayerNorm, window_size=8,
				 attn_ratio=0., attn_loc='last', conv_type=None):

		super().__init__()
		self.dim = dim
		self.depth = depth

		attn_depth = attn_ratio * depth

		if attn_loc == 'last':
			use_attns = [i >= depth-attn_depth for i in range(depth)]
		elif attn_loc == 'first':
			use_attns = [i < attn_depth for i in range(depth)]
		elif attn_loc == 'middle':
			use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

		# build blocks
		self.blocks = nn.ModuleList([
			TransformerBlock(network_depth=network_depth,
							 dim=dim, 
							 num_heads=num_heads,
							 mlp_ratio=mlp_ratio,
							 norm_layer=norm_layer,
							 window_size=window_size,
							 shift_size=0 if (i % 2 == 0) else window_size // 2,
							 use_attn=use_attns[i], conv_type=conv_type)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x


class PatchUnEmbed(nn.Module):
	def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = 1

		self.proj = nn.Sequential(
			nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
					  padding=kernel_size//2, padding_mode='reflect'),
			nn.PixelShuffle(patch_size)
		)

	def forward(self, x):
		x = self.proj(x)
		return x




class MAMFNet(nn.Module):
	def __init__(self, in_chans=3, out_chans=4, window_size=8,
				 embed_dims=[48, 96, 192, 384],
				 mlp_ratios=[2., 4., 4., 2.],
				 depths=[8, 8, 8, 4],
				 num_heads=[2, 4, 6, 1],
				 attn_ratio=[1/4, 1/2, 3/4, 1/2],
				 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
				 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
		super(MAMFNet, self).__init__()

		# setting
		self.patch_size = 4
		self.window_size = window_size
		self.mlp_ratios = mlp_ratios

		# split image into non-overlapping patches
		self.patch_embed = PatchEmbed(
			patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)



		self.addrelu = nn.ReLU()

        #CNNblock
		self.E_block1 = nn.Sequential(
			nn.Conv2d(3, 24, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(24, 24, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))	
		
		self.E_block2 = nn.Sequential(
			nn.Conv2d(24, 48, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(48, 48, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))	
		
		self.E_block3 = nn.Sequential(
			nn.Conv2d(48, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(96, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))	
		
		self.E_block4 = nn.Sequential(
			nn.Conv2d(96, 192, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 192, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))	

		self.D_block1 = nn.Sequential(
			nn.Conv2d(192, 96, 5, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(96, 96, 5, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.UpsamplingBilinear2d(scale_factor=2)
		)

		self.D_block2 = nn.Sequential(
			nn.Conv2d(96, 48, 5, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(48, 48, 5, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.UpsamplingBilinear2d(scale_factor=2)
		)

		self.D_block3 = nn.Sequential(
			nn.Conv2d(48, 24, 5, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(24, 24, 5, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.UpsamplingBilinear2d(scale_factor=2)
		)

		self.D_block4 = nn.Sequential(
			nn.Conv2d(24, 3, 5, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(3, 3, 5, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.UpsamplingBilinear2d(scale_factor=2)
		)				

		#new upsample 
		self.formerupsample1 = nn.Sequential(nn.Conv2d(embed_dims[0],embed_dims[0]*2,kernel_size=3,stride=1,padding=1),nn.PixelShuffle(2))
		self.formerupsample2 = nn.Sequential(nn.Conv2d(embed_dims[1],embed_dims[1]*2,kernel_size=3,stride=1,padding=1),nn.PixelShuffle(2))
		self.formerupsample3 = nn.Sequential(nn.Conv2d(embed_dims[2],embed_dims[2]*2,kernel_size=3,stride=1,padding=1),nn.PixelShuffle(2))
		self.formerupsample4 = nn.Sequential(nn.Conv2d(embed_dims[3],embed_dims[3]*2,kernel_size=3,stride=1,padding=1),nn.PixelShuffle(2))												




		self.upsmaple1 = nn.ConvTranspose2d(in_channels = 48, out_channels = 24 , kernel_size = 4, stride =2 ,padding = 1)
		self.upsmaple2 = nn.ConvTranspose2d(in_channels = 96, out_channels = 48 , kernel_size = 4, stride =2 ,padding = 1)
		self.upsmaple3 = nn.ConvTranspose2d(in_channels = 192, out_channels = 96 , kernel_size = 4, stride =2 ,padding = 1)
		self.upsmaple4 = nn.ConvTranspose2d(in_channels = 384, out_channels = 192 , kernel_size =4, stride =2 ,padding = 1)
								
			

		self.en_conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1)
		self.en_conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=2, padding=1)
		self.en_conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=2, padding=1)
		self.en_conv4 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=2, padding=1)						
        #decoder_block


		self.de_conv1 = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=4, stride=2, padding=1)
		self.de_conv2 = nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1)
		self.de_conv3 = nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1)
		self.de_conv4 = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=5, stride=1, padding=2)        


		self.attention1 = ChannelPixelAttention(in_channels= 24)
		self.attention2 = ChannelPixelAttention(in_channels= 48)
		self.attention3 = ChannelPixelAttention(in_channels= 96)
		self.attention4 = ChannelPixelAttention(in_channels= 192)
         

		# transformer
		self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
					   			 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
					   			 norm_layer=norm_layer[0], window_size=window_size,
					   			 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

		self.patch_merge1 = PatchEmbed(
			patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
		
		self.patch_merge2 = PatchEmbed(
			patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

		self.patch_merge3 = PatchEmbed(
			patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
					



		self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
								 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
								 norm_layer=norm_layer[1], window_size=window_size,
								 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

		self.patch_merge2 = PatchEmbed(
			patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])



		self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
								 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
								 norm_layer=norm_layer[2], window_size=window_size,
								 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

		self.patch_split1 = PatchUnEmbed(
			patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

		self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
								 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
								 norm_layer=norm_layer[3], window_size=window_size,
								 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])


		self.patch_unembed = PatchUnEmbed(
			patch_size=1, out_chans=out_chans, embed_dim=embed_dims[3], kernel_size=3)





	def check_image_size(self, x):
		# NOTE: for I2I test
		_, _, h, w = x.size()
		mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
		mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
		x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x



	def forward_features(self, x):
		
##encoder 
        #layer one 
		enc_x1 = self.en_conv1(x)
		enc_x1 = self.addrelu(enc_x1)
		x1 = self.patch_embed(x)	
		x1 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
		tran_x = self.layer1(x1)
		tran_x1 = self.formerupsample1(tran_x)
		f_add = enc_x1 + tran_x1			
		AT_map = self.attention1(f_add)				
		f_multi = enc_x1 * AT_map
	


        #layer two
		enc_x2 = self.en_conv2(f_multi)
		enc_x2 = self.addrelu(enc_x2)
		tran_x = self.patch_merge1(tran_x)
		tran_x = self.layer2(tran_x)
		tran_x2 = self.formerupsample2(tran_x)	
		f_add = enc_x2 + tran_x2
		AT_map = self.attention2(f_add)
		f_multi = enc_x2 * AT_map


		#layer three
		enc_x3 = self.en_conv3(f_multi)
		enc_x3 = self.addrelu(enc_x3)			
		tran_x = self.patch_merge2(tran_x)		
		tran_x = self.layer3(tran_x)	
		tran_x3 = self.formerupsample3(tran_x)		
		f_add = enc_x3 + tran_x3
		AT_map = self.attention3(f_add)
		f_multi = enc_x3 * AT_map	



		#layer four
		enc_x4 = self.en_conv4(f_multi)
		enc_x4 = self.addrelu(enc_x4)		
		tran_x = self.patch_merge3(tran_x)
		tran_x = self.layer4(tran_x)
		tran_x4 = self.formerupsample4(tran_x)		
		f_add = enc_x4 + tran_x4
		AT_map = self.attention4(f_add)
		f_multi = enc_x4 * AT_map	


##decoder
		f_add = f_multi + tran_x4
		dec1 = self.de_conv1(f_add)
		dec1 = enc_x3 + dec1
		dec2 = self.de_conv2(dec1)
		dec2 = enc_x2 + dec2
		dec3 = self.de_conv3(dec2)
		dec3 = enc_x1 + dec3
		dec4 = self.de_conv4(dec3)

		return dec4
		
	def forward(self, x):
		# H, W = x.shape[2:]
		# x = self.check_image_size(x)

		feat = self.forward_features(x)
		# K, B = torch.split(feat, (1, 3), dim=1)

		# x = K * x - B + x
		# x = x[:, :, :H, :W]
		return feat



