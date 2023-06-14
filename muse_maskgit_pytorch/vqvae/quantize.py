import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    def __init__(self, n_e, vq_embed_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.normal_()

    def forward(self, z):
        z = F.normalize(z, p=2, dim=-1)
        z_flattened = z.view(-1, self.vq_embed_dim)
        embed_norm = F.normalize(self.embedding.weight, p=2, dim=-1)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embed_norm**2, dim=1)
            - 2 * torch.einsum("bd,nd->bn", z_flattened, embed_norm)
        )

        encoding_indices = torch.argmin(d, dim=1).view(*z.shape[:-1])
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_q = F.normalize(z_q, p=2, dim=-1)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices

    def get_codebook_entry(self, indices):
        z_q = self.embedding(indices)
        z_q = F.normalize(z_q, p=2, dim=-1)

        return z_q
