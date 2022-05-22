import torch
import torch.nn as nn
import torch.nn.functional as F

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

class RAE(nn.Module):
    def __init__(self, input_size, latent_size, max_len, num_layers, r_model):
        super(RAE, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.linear1 = nn.Sequential(
            nn.Linear(input_size*max_len, latent_size*2 * max_len),
            nn.LeakyReLU(0.9)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*max_len, latent_size),
            nn.LeakyReLU(0.9)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(latent_size, latent_size*max_len),
            nn.LeakyReLU(0.9)
        )
        self.linear6 = nn.Sequential(
            nn.Linear(latent_size*2 * max_len, input_size*max_len),
            nn.LeakyReLU(0.9)
        )
        if r_model == 'LSTM':
            self.rnn2 = nn.LSTM(latent_size*2, latent_size, num_layers, batch_first=True)
            self.rnn5 = nn.LSTM(latent_size, latent_size*2, num_layers, batch_first=True)
        elfi r_model == 'GRU':
            self.rnn2 = nn.GRU(latent_size*2, latent_size, num_layers, batch_first=True)
            self.rnn5 = nn.GRU(latent_size, latent_size*2, num_layers, batch_first=True)
        else:
            print('No RNN Model')

    def forward(self, x):
        x = self.linear1(x.view(x.shape[0], -1))
        x, _ = self.rnn2(x.view(x.shape[0], self.max_len, -1))
        x = self.linear3(x.reshape(x.shape[0], -1))
        x = self.linear4(x)
        x, _ = self.rnn5(x.view(x.shape[0], self.max_len, -1))
        x = self.linear6(x.reshape(x.shape[0], -1))

        return x.view(-1, self.max_len, self.input_size)

class CRAE(nn.Module):
    def __init__(self, input_size, latent_size, max_len, num_layers, r_model):
        super(CRAE, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, latent_size*2, 3, 2, 1),
            nn.LeakyReLU(0.9)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*(max_len//2), latent_size),
            nn.LeakyReLU(0.9)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(latent_size, latent_size*(max_len//2)),
            nn.LeakyReLU(0.9)
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose1d(latent_size*2, input_size, 2, 2),
            nn.LeakyReLU(0.9)
        )
        if r_model == 'LSTM':
            self.rnn2 = nn.LSTM(latent_size*2, latent_size, num_layers, batch_first=True)
            self.rnn5 = nn.LSTM(latent_size, latent_size*2, num_layers, batch_first=True)
        elfi r_model == 'GRU':
            self.rnn2 = nn.GRU(latent_size*2, latent_size, num_layers, batch_first=True)
            self.rnn5 = nn.GRU(latent_size, latent_size*2, num_layers, batch_first=True)
        else:
            print('No RNN Model')

    def forward(self, x):
        x = self.conv1(x.permute(0,2,1))
        x, _ = self.rnn2(x.permute(0,2,1))
        x = self.linear3(x.reshape(x.shape[0], -1))
        x = self.linear4(x)
        x, _ = self.rnn5(x.view(x.shape[0], (self.max_len//2), -1))
        x = self.conv6(x.permute(0,2,1))
        x = x.permute(0,2,1)

        return x.view(-1, self.max_len, self.input_size)

class CRAE2(nn.Module):
    def __init__(self, input_size, latent_size, max_len, num_layers, r_model):
        super(CRAE2, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, latent_size*2, 3, 1, 1),
            nn.LeakyReLU(0.9)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*(max_len//2), latent_size),
            nn.LeakyReLU(0.9)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(latent_size, latent_size*(max_len//2)),
            nn.LeakyReLU(0.9)
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose1d(latent_size*2, input_size, 2, 2),
            nn.LeakyReLU(0.9)
        )
        if r_model == 'LSTM':
            self.rnn2 = nn.LSTM(latent_size*2, latent_size, num_layers, batch_first=True)
            self.rnn5 = nn.LSTM(latent_size, latent_size*2, num_layers, batch_first=True)
        elfi r_model == 'GRU':
            self.rnn2 = nn.GRU(latent_size*2, latent_size, num_layers, batch_first=True)
            self.rnn5 = nn.GRU(latent_size, latent_size*2, num_layers, batch_first=True)
        else:
            print('No RNN Model')

    def forward(self, x):
        x = max_pool1d(self.conv1(x.permute(0,2,1)), 2)
        x, _ = self.rnn2(x.permute(0,2,1))
        x = self.linear3(x.reshape(x.shape[0], -1))
        x = self.linear4(x)
        x, _ = self.rnn5(x.view(x.shape[0], (self.max_len//2), -1))
        x = self.conv6(x.permute(0,2,1))
        x = x.permute(0,2,1)

        return x.view(-1, self.max_len, self.input_size)




