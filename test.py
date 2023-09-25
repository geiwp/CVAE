import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
# 定义CVAE模型
class CVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(CVAE, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # 输出均值和对数方差
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 128),
            nn.ReLU(),
            nn.Linear(128, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()  # 输出图像像素值
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, y):
        x = x.view(-1, 784)
        # 编码
        enc = self.encoder(x)
        mu, logvar = enc[:, :latent_dim], enc[:, latent_dim:]
        z = self.reparameterize(mu, logvar)
        # 添加条件信息
        z = torch.cat((z, y), dim=1)
        # 解码
        return self.decoder(z), mu, logvar

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 准备数据集
batch_size = 128
latent_dim = 2

transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
model = CVAE(latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data
        labels = torch.eye(10)[labels]  # 转换为独热编码
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Epoch: {} Average loss: {:.4f}'.format(epoch+1, train_loss / len(train_loader.dataset)))

# 随机生成标签信息
random_labels = torch.randint(0, 10, (10,))
random_labels_onehot = torch.zeros(10, 10)
random_labels_onehot.scatter_(1, random_labels.unsqueeze(1), 1)

# 在潜在空间中生成随机样本
random_samples = torch.randn(10, latent_dim)

# 添加标签信息
random_samples_with_labels = torch.cat((random_samples, random_labels_onehot), dim=1)

# 使用解码器生成图像
with torch.no_grad():
    fig, axes = plt.subplots(8, 10, figsize=(15, 15))
# 循环生成图像并在每个子图上绘制
    for i in range(8):
        
        generated_images = model.decoder(random_samples_with_labels)
        for j in range(10):
            axes[i,j].imshow(generated_images[j].view(28, 28).numpy(), cmap='gray')
            axes[i,j].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

