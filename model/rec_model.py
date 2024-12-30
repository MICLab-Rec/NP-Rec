import torch
import torch.nn as nn
import torch.nn.functional as F
from Attempt.model.classifier import Classifier
from Attempt.model.seq_model import SSM
from Attempt.model.np_model import DeterministicNet, LatentNet, Decoder

class NPRec(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_latent):
        super(NPRec, self).__init__()
        self.num_class = vocab_size
        self.num_latent = num_latent
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.norm_embedding = nn.LayerNorm(embed_dim)
        self.drop_embedding = nn.Dropout(p=0.3)

        d_model = embed_dim
        self.seq_encoder = SSM(d_model)

        self.deterministic_encoder = DeterministicNet(d_model, d_model, self.num_class)
        self.latent_encoder = LatentNet(d_model, d_model, self.num_class, self.num_latent)
        self.decoder = Decoder(d_model * 3, d_model)

        self.classifier = Classifier(d_model, vocab_size, softmax=True)

    def encode_sequence(self, seq):
        seq = self.embedding(seq)
        seq = self.norm_embedding(seq)
        seq = self.drop_embedding(seq)
        # seq = self.seq_encoder(seq)
        return seq

    def train_model(self, src_seq, trg_seq):
        # [b, n, d]
        x = self.encode_sequence(src_seq)
        b, n, d = x.size()
        # [b, n/2, d]
        x_c, x_t = x.chunk(2, dim=1)
        y_c, y_t = trg_seq.chunk(2, dim=1)

        # [b, d]
        determ_repr = self.deterministic_encoder(x_c, y_c)
        # [b, t, d]
        prior_mu, prior_log_var, prior = self.latent_encoder(x_c, y_c)
        posterior_mu, posterior_log_var, posterior = self.latent_encoder(x_t, y_t)
        _, t, _ = posterior.size()
        
        determ_repr = determ_repr.unsqueeze(1).unsqueeze(1).expand(-1, n, t, d)
        posterior = posterior.unsqueeze(1).expand(-1, n, -1, -1)
        x = x.unsqueeze(2).expand(-1, -1, t, -1)
        
        decoder_input = torch.cat([determ_repr, posterior, x], dim=-1)
        decoder_output = self.decoder(decoder_input)
        
        probs = self.classifier(decoder_output, mode='training')
        kl = self.kl_divergence(prior_mu, prior_log_var, posterior_mu, posterior_log_var)
        return probs, kl
    
    def inference(self, src_seq, ctx_seq, dsz):
        # [b, n, d]
        x = self.encode_sequence(src_seq)
        b, n, d = x.size()
        # [b, 1, d]
        x_t = x[torch.arange(b), dsz - 1, :].unsqueeze(1).unsqueeze(1)
        x[torch.arange(b), dsz - 1, :] = 0.0
        x_c = x
        y_c = ctx_seq
        # [b, 1, 1, d]
        determ_repr = self.deterministic_encoder(x_c, y_c).unsqueeze(1).unsqueeze(1)
        # [b, 1, t, d]
        _, _, prior = self.latent_encoder(x_c, y_c)

        x_t = x_t.expand(-1, -1, self.num_latent, -1)
        determ_repr = determ_repr.expand(-1, 1, self.num_latent, -1)
        prior = prior.unsqueeze(1)
        decoder_input = torch.cat([determ_repr, prior, x_t], dim=-1)
        decoder_output = self.decoder(decoder_input)

        probs = self.classifier(decoder_output, mode='inference').squeeze(1)

        return probs


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def kl_divergence(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / torch.exp(prior_var) - 1. + (
                    prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div
