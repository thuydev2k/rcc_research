import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

class CNN_Extractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(

            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.downscale = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.downscale(x1)
        x2 = self.conv2(x1)
        x2 = self.downscale(x2)
        x3 = self.conv3(x2)
        x3 = self.downscale(x3)
        return x1, x2, x3



#######################################################################################################################################

       
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=256, patch_size=2, emb_dim=512, img_size=(64, 64)):
        super().__init__()
        
        self.patch_size = patch_size
        self.image_size = img_size
        self.embed_size = emb_dim

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, self.embed_size, kernel_size=self.patch_size, stride=self.patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        # Initialize the class token with a shape of (1, 1, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_size))

        # Calculate the number of patches
        num_patches = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)

        # Initialize positional embeddings
        self.positions = nn.Parameter(torch.randn(num_patches + 1, self.embed_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class External_Attention(nn.Module):
    def __init__(self, h=8):
        super().__init__()

        self.m_k = nn.Linear(64, 64)
        self.m_v = nn.Linear(64, 64)
        self.act_layer = nn.Softmax(dim=2)
        self.layer_norm = nn.LayerNorm(64)
        self.head_num = h

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.head_num, C // self.head_num)
        x = x.permute(0, 2, 1, 3)
        x = self.m_k(x)
        x = self.act_layer(x)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1, 3)
#        x = x.reshape(B, N, C)
        return x


class Token_Selector(nn.Module):
    def __init__(self, num_heads, extension_ratio):
        super().__init__()
        
        self.num_heads = num_heads
        self.extension_ratio = extension_ratio
        self.conv1d1 = nn.Conv1d(self.num_heads, self.num_heads * extension_ratio, 1)
        self.headwise_conv = nn.Conv1d(self.num_heads * extension_ratio, self.num_heads * 
                                       extension_ratio, kernel_size=3, padding=1, groups=self.num_heads * extension_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.conv1d2 = nn.Conv1d(self.num_heads * extension_ratio, self.num_heads, 1)

    def forward(self, x):
        B, seq_len, heads, dim_per_head = x.shape
        x = x.reshape(B, heads, seq_len * dim_per_head)  #(B, extended_num_heads, seq_len * dim per head)
        x = self.conv1d1(x)
        x = self.headwise_conv(x)
        x = self.relu(x)
        x = self.conv1d2(x)
        x = x.reshape(B, seq_len, dim_per_head * self.num_heads)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = External_Attention()
        self.token_selector = Token_Selector(num_heads=8, extension_ratio=6)
        self.layernorm1 = nn.LayerNorm(512)
        self.layernorm2 = nn.LayerNorm(512)
        self.mlp = MLP(512, 512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layernorm1(x)
        _x = self.attention(x)
        _x = self.token_selector(_x)
        _x = self.dropout(_x)
        x = x + _x
        _x = self.layernorm2(x)
        _x = self.mlp(_x)
        x = x + _x

        return x


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock() for _ in range(12)]
        )

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class  ViT(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.patchembed = PatchEmbedding()
        self.dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patchembed(x)
        x = self.dropout(x)
        x = self.transformer(x)[:, 1:, :]
        x = x.permute(0, 2, 1)
        x = x.view(B, 512, 32, 32)
        
        return x

#####################################################################################################################################################################
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.cnn = CNN_Extractor()
        self.vit = ViT()
        self.norm = nn.BatchNorm2d(512)
      
       
    def forward(self, x):
        
        x1, x2, x3 = self.cnn(x)
        x4 = self.vit(x3)
        x4 = self.norm(x4)
        return x1, x2, x3, x4
  
#########################################################################################################################################################

class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, out_classes=4):
        super().__init__()

        # x4 upsampled 512 + x3 256 = 768
        self.decoder1 = DecoderBottleneck(768, 128)

        # decoder1 output 128 + x2 128 = 256
        self.decoder2 = DecoderBottleneck(256, 64)

        # decoder2 output 64 + x1 64 = 128
        self.decoder3 = DecoderBottleneck(128, 64)

        # no skip here, only 64 input
        self.decoder4 = DecoderBottleneck(64, 32)

        self.conv1 = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)
        return x
		
		#######################################################################################################################################################


class TransUNet_Lite(nn.Module):
    def __init__(self, out_classes=4):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder(out_classes=out_classes)
        # self.head = nn.Conv2d(32, out_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, seq):
        
       
        x1, x2, x3, x4 = self.encoder(seq)          
        x = self.decoder(x4, x1, x2, x3)
        # x = self.head(x)
        
        return  x

# model =  TransUNet_Lite().cuda()

#######################################################################

	