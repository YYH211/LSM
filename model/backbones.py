import numpy as np
import os.path
import os

import gdown
import timm
import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed = ConvDownSampler(in_chans, 192, 2)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    # def initialize_weights(self):
    #     # initialization
    #     # initialize (and freeze) pos_embed by sin-cos embedding
    #     pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
    #                                         cls_token=True)
    #     self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    #
    #     decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
    #                                                 int(self.patch_embed.num_patches ** .5), cls_token=True)
    #     self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    #
    #     # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    #     w = self.patch_embed.proj.weight.data
    #     torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    #
    #     # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    #     torch.nn.init.normal_(self.cls_token, std=.02)
    #     torch.nn.init.normal_(self.mask_token, std=.02)
    #
    #     # initialize nn.Linear and nn.LayerNorm
    #     self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

__all__ = ["xcit_nano", "xcit_tiny12"]

file_ids = {
    "xcit_nano": "1c347oGdOd2vQD3vzTqKIv1rxXKfW1Ak6",
    "xcit_tiny12": "1DKd5E3WwEZxt99qCeSIzvgc1AWEEfdue",
}


class ConvDownSampler(nn.Module):
    def __init__(self, in_chans, embed_dim, ds_rate=16):
        super().__init__()
        ds_rate //= 2
        chan = embed_dim // ds_rate
        blocks = [nn.Conv1d(in_chans, chan, 5, 2, 2),
                  nn.BatchNorm1d(chan), nn.SiLU()]

        while ds_rate > 1:
            blocks += [
                nn.Conv1d(chan, 2 * chan, 5, 2, 2),
                nn.BatchNorm1d(2 * chan),
                nn.SiLU(),
            ]
            ds_rate //= 2
            chan = 2 * chan

        blocks += [
            nn.Conv1d(
                chan,
                embed_dim,
                1,
            )
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, X):
        return self.blocks(X)


class Chunker(nn.Module):
    def __init__(self, in_chans, embed_dim, ds_rate=16):
        super().__init__()
        self.embed = nn.Conv1d(in_chans, embed_dim // ds_rate, 7, padding=3)
        self.project = nn.Conv1d(
            (embed_dim // ds_rate) * ds_rate, embed_dim, 1)
        self.ds_rate = ds_rate

    def forward(self, X):
        X = self.embed(X)
        X = torch.cat(
            [
                torch.cat(torch.split(x_i, 1, -1), 1)
                for x_i in torch.split(X, self.ds_rate, -1)
            ],
            -1,
        )
        X = self.project(X)

        return X


class XCiT(nn.Module):
    def __init__(self, backbone, patch_len=12, in_chans=2, ds_rate=2, ds_method="downsample"):
        super().__init__()
        self.backbone = backbone
        W = backbone.num_features
        # self.grouper = nn.Conv1d(W, backbone.num_classes, 1)
        if ds_method == "downsample":
            self.backbone.patch_embed = ConvDownSampler(in_chans, W, ds_rate)
            test = torch.randn((2, in_chans, patch_len))
            shape = self.backbone.patch_embed(test)
        else:
            self.backbone.patch_embed = Chunker(in_chans, W, ds_rate)
            test = torch.randn((2, in_chans, patch_len))
            shape = self.backbone.patch_embed(test)

        Y = shape.shape[-1] + 1
        # Y = shape.shape[-1]
        self.Flatten = nn.Flatten()
        self.linear = nn.Linear(W, 10)
        self.linear_1 = nn.Linear(Y, 10)

    def forward(self, x):
        mdl = self.backbone
        B = x.shape[0]
        x = self.backbone.patch_embed(x)

        Hp, Wp = x.shape[-1], 1
        pos_encoding = (
            mdl.pos_embed(B, Hp, Wp).reshape(B, -1, Hp).permute(0, 2, 1).half()
        )
        x = x.transpose(1, 2) + pos_encoding

        for blk in mdl.blocks:
            x = blk(x, Hp, Wp)

        cls_tokens = mdl.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in mdl.cls_attn_blocks:
            x = blk(x)
        x = mdl.norm(x)

        # x = self.linear_1(x.transpose(1, 2))
        # x = self.linear(x)

        x = self.Flatten(x)
        # feature = x
        # feature = x.transpose(1, 2)[:, :, :1].squeeze()

        return x

def xcit_nano(
        pretrained: bool = False,
        path: str = "xcit_nano.pt",
        num_classes: int = 53,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.3,
        in_chans: int = 2,
):
    """Constructs a XCiT-Nano architecture from
    `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

    Args:
        pretrained (bool):
            If True, returns a model pre-trained on Sig53

        path (str):
            Path to existing model or where to download checkpoint to

        num_classes (int):
            Number of output classes; if loading checkpoint and number does not
            equal 53, final layer will not be loaded from checkpoint

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """
    model_exists = os.path.exists(path)
    if not model_exists and pretrained:
        file_id = file_ids["xcit_nano"]
        dl = gdown.download(id=file_id, output=path)
    mdl = XCiT(
        timm.create_model(
            'xcit_nano_12_p16_224',
            num_classes=53,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            in_chans=in_chans,
        ),
    )
    if pretrained:
        mdl.load_state_dict(torch.load(path))
    if num_classes != 53:
        mdl.classifier = nn.Linear(mdl.classifier.in_features, num_classes)
    return mdl


def xcit_tiny12(
        patch_len: int = 0,
        pretrained: bool = False,
        path: str = "xcit_tiny12.pt",
        num_classes: int = 53,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.3,
        in_chans: int = 2,
):
    """Constructs a XCiT-Tiny12 architecture from
    `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

    Args:
        pretrained (bool):
            If True, returns a model pre-trained on Sig53

        path (str):
            Path to existing model or where to download checkpoint to

        num_classes (int):
            Number of output classes; if loading checkpoint and number does not
            equal 53, final layer will not be loaded from checkpoint

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """
    # model_exists = os.path.exists(path)
    # if not model_exists and pretrained:
    #     file_id = file_ids["xcit_tiny12"]
    #     dl = gdown.download(id=file_id, output=path)
    mdl = XCiT(
        timm.create_model(
            'xcit_tiny_12_p16_224',
            num_classes=53,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,

        )
        ,
        patch_len=patch_len,
        in_chans=in_chans,
    )
    # if pretrained:
    #     mdl.load_state_dict(torch.load(path), strict=False)
    # if num_classes != 53:
    #     mdl.grouper = nn.Conv1d(mdl.grouper.in_channels, num_classes, 1)
    return mdl


__all__ = ["efficientnet_b0", "efficientnet_b2", "efficientnet_b4"]

# file_ids = {
#     "efficientnet_b0": "1ZQIBRZJiwwjeP4rB7HxxFzFro7RbxihG",
#     "efficientnet_b2": "1yaPZS5bbf6npHfUVdswvUnsJb8rDHlaa",
#     "efficientnet_b4": "1KCoLY5X0rIc_6ArmZRdkxZOOusIHN6in",
# }


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.SiLU,
        gate_fn=torch.sigmoid,
        divisor=1,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        reduced_chs = reduced_base_chs
        self.conv_reduce = nn.Conv1d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv1d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2,), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class FastGlobalAvgPool1d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool1d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return (
                x.view(x.size(0), x.size(1), -
                       1).mean(-1).view(x.size(0), x.size(1), 1)
            )


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.1):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


def replace_bn(parent):
    for n, m in parent.named_children():
        if type(m) is nn.BatchNorm2d:
            setattr(
                parent,
                n,
                GBN(m.num_features),
            )
        else:
            replace_bn(m)


def replace_se(parent):
    for n, m in parent.named_children():
        if type(m) is timm.models.efficientnet_blocks.SqueezeExcite:
            setattr(
                parent,
                n,
                SqueezeExcite(
                    m.conv_reduce.in_channels,
                    reduced_base_chs=m.conv_reduce.out_channels,
                ),
            )
        else:
            replace_se(m)


def replace_conv_effnet(parent, ds_rate):
    for n, m in parent.named_children():
        if type(m) is nn.Conv2d:
            if ds_rate == 2:
                setattr(
                    parent,
                    n,
                    nn.Conv1d(
                        m.in_channels,
                        m.out_channels,
                        kernel_size=m.kernel_size[0],
                        stride=m.stride[0],
                        padding=m.padding[0],
                        bias=m.kernel_size[0],
                        groups=m.groups,
                    ),
                )
            else:
                setattr(
                    parent,
                    n,
                    nn.Conv1d(
                        m.in_channels,
                        m.out_channels,
                        kernel_size=m.kernel_size[0] if m.kernel_size[0] == 1 else 5,
                        stride=m.stride[0] if m.stride[0] == 1 else ds_rate,
                        padding=m.padding[0] if m.padding[0] == 0 else 2,
                        bias=m.kernel_size[0],
                        groups=m.groups,
                    ),
                )
        else:
            replace_conv_effnet(m, ds_rate)


def create_effnet(network, ds_rate=2):
    replace_se(network)
    replace_bn(network)
    replace_conv_effnet(network, ds_rate)
    network.global_pool = FastGlobalAvgPool1d(flatten=True)
    return network


def efficientnet_b0(
    pretrained: bool = False,
    path: str = "efficientnet_b0.pt",
    num_classes: int = 53,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
):
    """Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on Sig53

        path (str): 
            Path to existing model or where to download checkpoint to

        num_classes (int): 
            Number of output classes; if loading checkpoint and number does not
            equal 53, final layer will not be loaded from checkpoint

        drop_path_rate (float): 
            Drop path rate for training

        drop_rate (float): 
            Dropout rate for training

    """
    model_exists = os.path.exists(path)
    if not model_exists and pretrained:
        file_id = file_ids["efficientnet_b0"]
        dl = gdown.download(id=file_id, output=path)
    mdl = create_effnet(
        timm.create_model(
            'efficientnet_b0',
            num_classes=0,
            in_chans=2,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
    )
    return mdl


def efficientnet_b2(
    pretrained: bool = False,
    path: str = "efficientnet_b2.pt",
    num_classes: int = 53,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
):
    """Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool):
            If True, returns a model pre-trained on Sig53

        path (str):
            Path to existing model or where to download checkpoint to

        num_classes (int):
            Number of output classes; if loading checkpoint and number does not
            equal 53, final layer will not be loaded from checkpoint

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """
    model_exists = os.path.exists(path)
    if not model_exists and pretrained:
        file_id = file_ids["efficientnet_b2"]
        dl = gdown.download(id=file_id, output=path)
    mdl = create_effnet(
        timm.create_model(
            'efficientnet_b2',
            num_classes=0,
            in_chans=2,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
    )
    return mdl


def efficientnet_b4(
    pretrained: bool = False,
    path: str = "efficientnet_b4.pt",
    num_classes: int = 53,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
):
    """Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool):
            If True, returns a model pre-trained on Sig53

        path (str):
            Path to existing model or where to download checkpoint to

        num_classes (int):
            Number of output classes; if loading checkpoint and number does not
            equal 53, final layer will not be loaded from checkpoint

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """
    model_exists = os.path.exists(path)
    if not model_exists and pretrained:
        file_id = file_ids["efficientnet_b4"]
        dl = gdown.download(id=file_id, output=path)
    mdl = create_effnet(
        timm.create_model(
            'efficientnet_b4',
            num_classes=0,
            in_chans=2,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
    )
    return mdl


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_input = torch.rand(10, 2, 409).cuda()
    bm_2 = efficientnet_b4().cuda()
    bm_1 = CNN(128).cuda()
    bm_1_output = bm_2(test_input)
    # bm_2_output = bm_2(test_input)
    print('success')


if __name__ == '__main__':
    main()
