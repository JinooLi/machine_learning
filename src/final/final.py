# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# use pytorch
# use nvidia gpu

# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import torch.nn.functional as F
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 경로 설정
train_dir = os.path.join(".", "data", "train")
test_dir = os.path.join(".", "data", "test")

# 이미지 전처리를 위한 transform 정의
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# 모델 클래스들 정의
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + residual


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.prelu(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()

        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])

        self.upsampling = nn.Sequential(UpsampleBlock(64), UpsampleBlock(64))

        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.prelu(self.conv1(x))
        residual = x
        x = self.res_blocks(x)
        x = x + residual
        x = self.upsampling(x)
        return torch.tanh(self.conv2(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 64
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 4, 1, kernel_size=1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:36]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return F.mse_loss(sr_features, hr_features)


# Loss 함수들
def content_loss(gen_features, real_features):
    return F.mse_loss(gen_features, real_features)


def adversarial_loss(pred, target):
    return F.binary_cross_entropy(pred, target)


# Training step 함수
def train_step(real_hr, generator, discriminator, opt_g, opt_d, vgg_loss):
    batch_size = real_hr.size(0)

    real_lr = F.interpolate(real_hr, scale_factor=0.25, mode="bicubic")
    fake_sr = generator(real_lr)

    # Train Discriminator
    opt_d.zero_grad()
    real_pred = discriminator(real_hr)
    fake_pred = discriminator(fake_sr.detach())
    d_loss_real = adversarial_loss(real_pred, torch.ones_like(real_pred))
    d_loss_fake = adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
    d_loss = (d_loss_real + d_loss_fake) / 2
    d_loss.backward()
    opt_d.step()

    # Train Generator
    opt_g.zero_grad()
    fake_pred = discriminator(fake_sr)

    perception_loss = vgg_loss(fake_sr, real_hr)
    adversarial_g_loss = adversarial_loss(fake_pred, torch.ones_like(fake_pred))
    pixel_loss = F.mse_loss(fake_sr, real_hr)

    g_loss = perception_loss * 0.006 + adversarial_g_loss * 0.001 + pixel_loss
    g_loss.backward()
    opt_g.step()

    return d_loss.item(), g_loss.item()


# Hyperparameters
num_epochs = 100
criterion = nn.BCELoss()


def main():
    # 데이터 로더 생성
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
    )

    # 모델 초기화
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg_loss = VGGLoss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, _) in progress_bar:
            d_loss, g_loss = train_step(
                images.to(device),
                generator,
                discriminator,
                optimizer_G,
                optimizer_D,
                vgg_loss,
            )

            progress_bar.set_description(
                f"Epoch [{epoch+1}/{num_epochs}] d_loss: {d_loss:.4f} g_loss: {g_loss:.4f}"
            )

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                },
                f"checkpoint_epoch_{epoch+1}.pth",
            )


if __name__ == "__main__":
    main()
