import torch
import torch.nn as nn
import torch.nn.functional as F

class DeconvLayer(nn.Module):
    def __init__(self, c_in):
        super(DeconvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.deconv = nn.ConvTranspose1d(in_channels=c_in, out_channels=c_in, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 1))
        x = self.activation(x)
        x = self.deconv(x)
        x = x.transpose(1,2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Decoder(nn.Module):
    def __init__(self, attn_layers, deconv_layers=None, norm_layer=None):
        super(Decoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.deconv_layers = nn.ModuleList(deconv_layers) if deconv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.deconv_layers is not None:
            for attn_layer, deconv_layer in zip(self.attn_layers, self.deconv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = deconv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
