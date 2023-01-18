import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


class Generator(nn.Module):
    def __init__(self, noise_sz=100, feature_sz=32):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(noise_sz, feature_sz * 16, 4, 1, 0),
            self._block(feature_sz * 16, feature_sz * 8, 4, 2, 1),
            self._block(feature_sz * 8, feature_sz * 4, 4, 2, 1),
            self._block(feature_sz * 4, feature_sz * 2, 4, 2, 1),
            nn.ConvTranspose2d(feature_sz * 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, feature_sz=32):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, feature_sz, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            self._block(feature_sz, feature_sz * 2, 4, 2, 1),
            self._block(feature_sz * 2, feature_sz * 4, 4, 2, 1),
            self._block(feature_sz * 4, feature_sz * 8, 4, 2, 1),
            nn.Conv2d(feature_sz * 8, 1, 4, 2, 0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x):
        return self.disc(x)


def weights_init(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class GlobalGeneratorDiscriminator(nn.Module):
    def __init__(self, gen, disc):
        super().__init__()
        self.G = gen
        self.D = disc

    def model_train(self, dataloader, batch_sz, noise_sz, epochs=5, lr=3e-4):

        criterion = nn.BCELoss()
        optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        lossesG = []
        lossesD = []

        self.D.train()
        self.G.train()

        fixed_noise = torch.randn(32, noise_sz, 1, 1).to(device)
        writer_real = SummaryWriter(f"My_logs/real")
        writer_fake = SummaryWriter(f"My_logs/fake")
        writer_loss_d = SummaryWriter(f"My_logs/loss_d")
        writer_loss_g = SummaryWriter(f"My_logs/loss_g")
        step = 0

        for epoch in range(epochs):
            for i, (real_data, _) in enumerate(dataloader):
                # D train
                real_data = real_data.to(device)
                noise = torch.randn((batch_sz, noise_size, 1, 1)).to(device)
                fake = self.G(noise)

                D_output_real = self.D(real_data).reshape(-1)
                D_loss_real = criterion(D_output_real, torch.ones_like(D_output_real, device=device))
                D_output_fake = self.D(fake.detach()).reshape(-1)
                D_loss_fake = criterion(D_output_fake, torch.zeros_like(D_output_fake, device=device))

                D_loss = (D_loss_real + D_loss_fake) / 2
                self.D.zero_grad()
                D_loss.backward()
                optD.step()

                # G train
                output = self.D(fake).reshape(-1)
                G_loss = criterion(output, torch.ones_like(output))

                self.G.zero_grad()
                G_loss.backward()
                optG.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, epochs, i, len(dataloader), D_loss.item(), G_loss.item()))
                    with torch.no_grad():
                        fake = self.G(fixed_noise)
                        img_grid_real = torchvision.utils.make_grid(real_data[:32], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                        writer_loss_d.add_scalar("Loss disc", D_loss, global_step=step)
                        writer_loss_g.add_scalar("Loss gen", G_loss, global_step=step)
                        writer_real.add_image("Real", img_grid_real, global_step=step)
                        writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                    step += 1
                lossesD.append(D_loss)
                lossesG.append(G_loss)


img_size = 64

transforms = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(1)], [0.5 for _ in range(1)]
        ),
    ]
)

dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms, download=True
)

batch_size = 128
noise_size = 100
device = torch.device('cuda')
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch = next(iter(train_loader))
batch_data, batch_labels = batch[0].to(device), batch[1].to(device)
noise = torch.rand((batch_size, noise_size, 1, 1)).to(device)

G = Generator(feature_sz=64).to(device)
weights_init(G)
assert G(noise).shape == (batch_size, 1, img_size, img_size), "G shape out failed"

D = Discriminator(feature_sz=64).to(device)
weights_init(D)
assert D(batch_data).shape == (batch_size, 1, 1, 1), "D shape out failed"


model = GlobalGeneratorDiscriminator(G, D).to(device)
model.model_train(dataloader=train_loader, batch_sz=batch_size, noise_sz=noise_size, epochs=5)
