import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CONFIG = {
    "vocab_size": 50368,
    "embed_dim": 512,
    "base_channels": 512,
    "latent_dim": 4,
    "num_res_blocks": (3, 3),
}

class ChannelRMSNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.RMSNorm(num_channels, eps=eps)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = ChannelRMSNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = ChannelRMSNorm1d(out_channels)

        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        return out + residual

class Downsample1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.deconv(x)

class Encoder1d(nn.Module):
    def __init__(self, vocab_size, embed_dim=128,
                 base_channels=64, latent_dim=128,
                 num_res_blocks=(2, 2)):
        super().__init__()

        # text embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # use embedding dim as conv input channels
        in_channels = embed_dim
        layers = []

        for level, n_blocks in enumerate(num_res_blocks):
            out_channels = base_channels * (2 ** level)

            for _ in range(n_blocks):
                layers.append(ResBlock1d(in_channels, out_channels))
                in_channels = out_channels

            down_channels = out_channels * 2
            layers.append(Downsample1d(out_channels, down_channels))
            in_channels = down_channels

        self.final_channels = in_channels
        layers.append(ResBlock1d(self.final_channels, self.final_channels))

        self.blocks = nn.Sequential(*layers)
        self.fc_mu = nn.Conv1d(self.final_channels, latent_dim, 1)
        self.fc_logvar = nn.Conv1d(self.final_channels, latent_dim, 1)

    def forward(self, tokens):
        x = self.embedding(tokens).transpose(1, 2)
        h = self.blocks(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

class Decoder1d(nn.Module):
    def __init__(self, vocab_size, embed_dim=128,
                 base_channels=64, latent_dim=128,
                 num_res_blocks=(2, 2)):
        super().__init__()

        self.vocab_size = vocab_size

        self.num_down = len(num_res_blocks)
        self.final_channels = base_channels * (2 ** self.num_down)

        self.fc = nn.Conv1d(latent_dim, self.final_channels, 1)
        self.bottom_block = ResBlock1d(self.final_channels, self.final_channels)

        layers = []
        in_channels = self.final_channels

        for level in reversed(range(self.num_down)):
            n_blocks = num_res_blocks[level]

            up_channels = in_channels // 2
            layers.append(Upsample1d(in_channels, up_channels))
            in_channels = up_channels

            for _ in range(n_blocks):
                layers.append(ResBlock1d(in_channels, in_channels))

        self.blocks = nn.Sequential(*layers)

        # output logits for each vocab token
        self.out_conv = nn.Conv1d(in_channels, vocab_size, 1)

    def forward(self, z):
        h = self.bottom_block(self.fc(z))
        h = self.blocks(h)
        logits = self.out_conv(h)     # (B, vocab, L)
        return logits

class TextVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=128,
                 base_channels=64, latent_dim=128,
                 num_res_blocks=(2, 2)):
        super().__init__()
        self.encoder = Encoder1d(vocab_size, embed_dim, base_channels,
                                 latent_dim, num_res_blocks)
        self.decoder = Decoder1d(vocab_size, embed_dim, base_channels,
                                 latent_dim, num_res_blocks)
        self.vocab_size = vocab_size

        self.kl_loss_factor = 1e-6
        self.downsample_ratio = 2 ** len(num_res_blocks)

    def print_parameters(self):
        total = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"Total params: {total:.2f}M")

        encoder_params = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        print(f"Encoder params: {encoder_params:.2f}M")

        decoder_params = sum(p.numel() for p in self.decoder.parameters()) / 1e6
        print(f"Decoder params: {decoder_params:.2f}M")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tokens):
        mu, logvar = self.encoder(tokens)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

    def compute_loss(self, tokens):
        logits, mu, logvar = self.forward(tokens)

        recon_loss = F.cross_entropy(
            logits,
            tokens,
            reduction="mean"
        )

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + kl_loss * self.kl_loss_factor

        return total_loss, {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def generate(self, tokens):
        logits, mu, logvar = self.forward(tokens)
        tokens = logits.argmax(dim=1)
        return tokens

if __name__ == "__main__":
    vocab = MODEL_CONFIG["vocab_size"]
    seq_len = 64
    batch = 4

    x = torch.randint(0, vocab, (batch, seq_len))

    model = TextVAE(**MODEL_CONFIG)

    model.print_parameters()

    loss, loss_dict = model.compute_loss(x)
    print(loss_dict)

    generated = model.generate(x)
    print(generated.shape)
    print(generated)