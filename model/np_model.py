import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MLP, self).__init__()
        self.merge_layer = nn.Linear(input_dim, latent_dim, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(latent_dim, latent_dim, bias=True)

    def forward(self, x):
        x = self.merge_layer(x)
        x = self.act(x)
        x = self.fc(x)
        return x


class DeterministicNet(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = MLP(input_dim + num_classes, latent_dim)

    def forward(self, sample, label):
        mask = (label > 0).float().to(sample.device)
        mask = mask.sum(dim=1, keepdim=True)

        y = F.one_hot(label, num_classes=self.num_classes).float().to(sample.device)
        enc_input = torch.cat([sample, y], dim=-1)
        enc_output = self.mlp(enc_input)

        hidden = enc_output.sum(dim=1)
        hidden = hidden / mask
        # [b, d]
        return hidden


class LatentNet(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes, num_latent):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = MLP(input_dim + num_classes, latent_dim)
        self.mean_layer = MLP(latent_dim, latent_dim)
        self.log_var_layer = MLP(latent_dim, latent_dim)
        self.num_latent = num_latent

    def forward(self, sample, label):
        mask = (label > 0).float().to(sample.device)
        mask = mask.sum(dim=1, keepdim=True)

        y = F.one_hot(label, num_classes=self.num_classes).float().to(sample.device)
        enc_input = torch.cat([sample, y], dim=-1)
        enc_output = self.mlp(enc_input)

        hidden = enc_output.sum(dim=1)
        hidden = hidden / mask
        # [b, d]
        mean = self.mean_layer(hidden)
        log_var = self.log_var_layer(hidden)
        sigma = 0.1 + 0.9 * F.softplus(log_var)

        z = []
        for i in range(self.num_latent):
            z.append(self.sample(mean, sigma))
        # [b, t, d]
        z = torch.stack(z, dim=1)
        return mean, sigma, z

    def sample(self, mean, std):
        eps = torch.randn_like(std)
        sample = mean + std * eps
        return sample


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.mlp = MLP(input_dim, latent_dim)

    def forward(self, x):
        x = self.mlp(x)
        return x
