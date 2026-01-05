import torch.nn as nn
import torch.nn.functional as F
from .basic import BasicLayer, PatchUnEmbed, PatchEmbed
from timm.models.layers import trunc_normal_


class SwinTBlock(nn.Module):
    """
    Swin Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(SwinTBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim)

    def forward(self, x, x_size):
        res = self.residual_group(x, x_size)  # depth transformer blocks
        unbed = self.patch_unembed(res, x_size)

        conv = self.conv(unbed) + unbed
        embed = self.patch_embed(conv)

        return embed + x


class SwinTransformer(nn.Module):
    """
    Args:
        img_size (int): Image size. Default: 64.
        patch_size (int): Patch size. Default: 1.
        in_chans (int): Number of input channels. Default: 3.
        embed_dim (int): Patch embed channels. Default: 96.
        depth (list(int)): Depth of each layer. Default: [6, 6, 6, 6].
        num_heads (int): Number of attention heads. Default: 3.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        resi_connection (str): The convolutional block before residual connection. Default: '1conv'.
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96, depth=None, num_heads=3,
                 window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False, resi_connection='1conv'):
        super(SwinTransformer, self).__init__()

        if depth is None:
            depth = [6, 6, 6, 6]
        self.window_size = window_size

        # ----- Feature Extraction Before Transformer ----- #
        self.conv_first = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_second = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)

        # ----- Transformer Encoder ----- #
        self.num_layers = len(depth)
        # the number of layers should be 4n, 2n each for encoder and decoder
        assert self.num_layers % 4 == 0
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # Image => Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.patches_resolution = self.patch_embed.grid_size

        self.encoder = nn.ModuleList()
        for i in range(self.num_layers // 2):
            self.encoder.append(
                SwinTBlock(
                    dim=embed_dim,
                    input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                    depth=depth[i],
                    num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint, resi_connection=resi_connection,
                    img_size=img_size, patch_size=patch_size
                )
            )
        self.norm = norm_layer(embed_dim)

        # ----- Transformer Decoder ----- #
        self.decoder = nn.ModuleList()
        for i in range(self.num_layers // 2, self.num_layers):
            self.decoder.append(
                SwinTBlock(
                    dim=embed_dim,
                    input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                    depth=depth[i],
                    num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint, resi_connection=resi_connection,
                    img_size=img_size, patch_size=patch_size
                )
            )

        # Patch embedding => Image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )

        # conv layer after transformer
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)

        # ----- Output Convolution ----- #
        self.conv_out = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_img_size(self, x):
        H, W = x.shape[-2], x.shape[-1]
        # make sure the input image size can be divided by window size
        mod_pad_H = (self.window_size - H % self.window_size) % self.window_size
        mod_pad_W = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_W, 0, mod_pad_H), mode='reflect')  # (left, right, top, bottom)
        return x

    def Encoder(self, x):
        self.H, self.W = x.shape[-2], x.shape[-1]
        x = self.check_img_size(x)

        # TODO: make sure the input is in the range of [-1, 1]

        x = self.conv_first(x)
        self.x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x_embed = x

        for blk in self.encoder:
            x = blk(x, self.x_size)

        return x + x_embed

    def Decoder(self, x, H=None, W=None):
        if H is not None:
            self.H, self.W = H, W

        x_embed = x
        for blk in self.decoder:
            x = blk(x, (self.H, self.W))

        x = self.norm(x + x_embed)
        x = self.patch_unembed(x, self.x_size)

        x = self.conv_after(x)
        x = self.conv_out(x)

        return x

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


if __name__ == '__main__':
    import torch
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = SwinTransformer(
        img_size=48, patch_size=2, in_chans=1, embed_dim=180, num_heads=4, window_size=8
    ).cuda()
    x = torch.randn(1, 1, 48, 48).cuda()
    y = model(x)
    print(y.shape)
