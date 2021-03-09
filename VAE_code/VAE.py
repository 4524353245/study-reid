import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from dataset_loader import ImageDataset
import data_manager
import torchvision.transforms as T

# Device configuration
# os.environ['CUDA_VISBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = '/home/user/Desktop'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
image_size = 32*16*3
h_dim = 400
z_dim = 20
num_epochs = 15
batch_size = 128
learning_rate = 1e-3

# # MNIST dataset
# dataset = torchvision.datasets.MNIST(root='../../data',
#                                      train=True,
#                                      transform=transforms.ToTensor(),
#                                      download=True)

# # Data loader
# data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                           batch_size=batch_size, 
#                                           shuffle=True)

args_root =  '/home/user/Desktop'  
args_dataset =  'market1501'
args_split_id = 0
args_height = 32
args_width = 16
args_train_batch = 128
args_workers = 4
use_gpu = torch.cuda.is_available()
pin_memory = True if use_gpu else False

transform_train = T.Compose([
        T.Resize((args_height, args_width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

dataset = data_manager.init_img_dataset(
        root=args_root, name=args_dataset, split_id=args_split_id,
    )

data_loader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=args_train_batch, shuffle=True, num_workers=args_workers,
        pin_memory=pin_memory, drop_last=True,
    )
# for batch_idx, (imgs, pids, _) in enumerate(data_loader):
#     print('batch_idx:',batch_idx,'imgs:',imgs.size())

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=32*16*3, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# Start training
for epoch in range(num_epochs):
    for i, (x,pids, _) in enumerate(data_loader):
        # Forward pass
        x = x.to(device).view(-1, image_size)
        # print('x:',x.shape)
        x_reconst, mu, log_var = model(x)
        # print('mode:',x_reconst.shape)

        # Compute reconstruction loss and kl divergence
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        # reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        reconst_loss = F.mse_loss(x_reconst, x, size_average=False)

        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))

    with torch.no_grad():
        # Save the sampled images
        z = torch.randn(batch_size, z_dim).to(device)
        out = model.decode(z).view(-1, 3, 32, 16)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # Save the reconstructed images
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 3, 32, 16), out.view(-1, 3, 32, 16)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))
