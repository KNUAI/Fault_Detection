import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer, DeconvLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class SAAE(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(SAAE, self).__init__()
        self.seq_len = seq_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                DeconvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out, de_attns = self.decoder(enc_out, attn_mask=dec_self_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.seq_len:,:], attns
        else:
            return dec_out[:,-self.seq_len:,:] # [B, L, D]

class AE(nn.Module):
    def __init__(self, input_size, latent_size, max_len):
        super(AE, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.linear1 = nn.Sequential(
            nn.Linear(input_size*max_len, latent_size*4),
            nn.LeakyReLU(0.9)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(latent_size*4, latent_size*2),
            nn.LeakyReLU(0.9)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*2, latent_size),
            nn.LeakyReLU(0.9)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(latent_size, latent_size*2),
            nn.LeakyReLU(0.9)
        )
        self.linear5 = nn.Sequential(
            nn.Linear(latent_size*2, latent_size*4),
            nn.LeakyReLU(0.9)
        )
        self.linear6 = nn.Sequential(
            nn.Linear(latent_size*4, input_size*max_len),
            nn.LeakyReLU(0.9)
        )

    def forward(self, x):
        x = self.linear1(x.view(-1, self.input_size*self.max_len))
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)

        return x.view(-1, self.max_len, self.input_size)

class CAE(nn.Module):
    def __init__(self, input_size, latent_size, max_len):
        super(CAE, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, latent_size, 3, 2, 1),
            nn.LeakyReLU(0.9)
        )                        
        self.conv2 = nn.Sequential(
            nn.Conv1d(latent_size, latent_size*2, 3, 2, 1),
            nn.LeakyReLU(0.9)
        )                        
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*2 * (max_len//4), latent_size),
            nn.LeakyReLU(0.9)
        )                        
        self.linear4 = nn.Sequential(
            nn.Linear(latent_size, latent_size*2 * (max_len//4)),
            nn.LeakyReLU(0.9)
        )                        
        self.conv5 = nn.Sequential(
            nn.ConvTranspose1d(latent_size*2, latent_size, 2, 2),
            nn.LeakyReLU(0.9)
        )                        
        self.conv6 = nn.Sequential(
            nn.ConvTranspose1d(latent_size, input_size, 2, 2),
            nn.LeakyReLU(0.9)
        )

    def forward(self, x):
        x = self.conv1(x.permute(0,2,1))
        x = self.conv2(x)
        x = self.linear3(x.view(x.shape[0], -1))
        x = self.linear4(x)
        x = self.conv5(x.view(x.shape[0], -1, (self.max_len//4)))
        x = self.conv6(x)
        x = x.permute(0,2,1)
        
        return x.view(-1, self.max_len, self.input_size)

class CAE2(nn.Module):
    def __init__(self, input_size, latent_size, max_len):
        super(CAE2, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, latent_size, 3, 1, 1),
            nn.LeakyReLU(0.9)
        )                        
        self.conv2 = nn.Sequential(
            nn.Conv1d(latent_size, latent_size*2, 3, 1, 1),
            nn.LeakyReLU(0.9)
        )                        
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*2 * (max_len//4), latent_size),
            nn.LeakyReLU(0.9)
        )                        
        self.linear4 = nn.Sequential(
            nn.Linear(latent_size, latent_size*2 * (max_len//4)),
            nn.LeakyReLU(0.9)
        )                        
        self.conv5 = nn.Sequential(
            nn.ConvTranspose1d(latent_size*2, latent_size, 2, 2),
            nn.LeakyReLU(0.9)
        )                        
        self.conv6 = nn.Sequential(
            nn.ConvTranspose1d(latent_size, input_size, 2, 2),
            nn.LeakyReLU(0.9)
        )

    def forward(self, x):
        x = F.max_pool1d(self.conv1(x.permute(0,2,1)), 2)
        x = F.max_pool1d(self.conv2(x), 2)
        x = self.linear3(x.view(x.shape[0], -1))
        x = self.linear4(x)
        x = self.conv5(x.view(x.shape[0], -1, (self.max_len//4)))
        x = self.conv6(x)
        x = x.permute(0,2,1)
        
        return x.view(-1, self.max_len, self.input_size)

class LSTM(nn.Module):
    def __init__(self, input_size, latent_size, max_len, num_layers):
        super(LSTM, self).__init__()
        self.max_len = max_len
        self.rnn1 = nn.LSTM(input_size, latent_size*4, num_layers, batch_first=True)
        self.rnn2 = nn.LSTM(latent_size*4, latent_size*2, num_layers, batch_first=True)
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*2*max_len, latent_size),
            nn.LeakyReLU(0.9)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(latent_size, latent_size*2*max_len),
            nn.LeakyReLU(0.9)
        )
        self.rnn5 = nn.LSTM(latent_size*2, latent_size*4, num_layers, batch_first=True)
        self.rnn6 = nn.LSTM(latent_size*4, input_size, num_layers, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn1(x)
        _, (h,c) = self.rnn2(x)
        h = h[-1,:,:].unsqueeze(-2).repeat(1,self.max_len,1)
        x = self.linear3(h.reshape(h.shape[0], -1))
        x = self.linear4(x)
        x, _ = self.rnn5(x.view(x.shape[0], self.max_len, -1))
        out, _ = self.rnn6(x)  #out : (batch_size, sequence_len, hidden_size)

        return out

class GRU(nn.Module):
    def __init__(self, input_size, latent_size, max_len, num_layers):
        super(GRU, self).__init__()
        self.max_len = max_len
        self.rnn1 = nn.GRU(input_size, latent_size*4, self.num_layers, batch_first=True)
        self.rnn2 = nn.GRU(latent_size*4, latent_size*2, self.num_layers, batch_first=True)
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*2*max_len, latent_size),
            nn.LeakyReLU(0.9)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(latent_size, latent_size*2*max_len),
            nn.LeakyReLU(0.9)
        )
        self.rnn5 = nn.GRU(latent_size*2, latent_size*4, self.num_layers, batch_first=True)
        self.rnn6 = nn.GRU(latent_size*4, input_size, self.num_layers, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn1(x)
        _, h = self.rnn2(x)
        h = h[-1,:,:].unsqueeze(-2).repeat(1,self.max_len,1)
        x = self.linear3(h.reshape(h.shape[0], -1))
        x = self.linear4(x)
        x, _ = self.rnn5(x.view(x.shape[0], self.max_len, -1))
        out, _ = self.rnn6(x)  #out : (batch_size, sequence_len, hidden_size)

        return out




