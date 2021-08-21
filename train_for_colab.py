"""
注意：
1.
    datasets.py
    network.py
    utils.py
    are in
    /content/drive/MyDrive/MyCycleGAN/*.py
2.
    my dataset root in Colab:
    /content/datasets/trainA or /content/datasets/trainB
"""

###      cell1 BEGIN      ###

import itertools
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from drive.MyDrive.MyCycleGAN.network import Generator
from drive.MyDrive.MyCycleGAN.network import Discriminator
from drive.MyDrive.MyCycleGAN.utils import ImageBuffer
from drive.MyDrive.MyCycleGAN.datasets import ImageDataset
from drive.MyDrive.MyCycleGAN.utils import LogInfo

###      cell1 END      ###



###      cell2 BEGIN      ###

! /opt/bin/nvidia-smi

###      cell2 END      ###



###      cell3 BEGIN      ###

! unzip '/content/trainA.zip' -d '/content/'
! unzip '/content/trainB.zip' -d '/content/'

###      cell3 END      ###



###      cell4 BEGIN      ###

def init_weight(m):
  if isinstance(m, nn.Conv2d):
    nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
  elif isinstance(m, nn.BatchNorm2d):  # Does this necessary?
    nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
    nn.init.constant_(m.bias.data, val=0.0)

root = 'datasets/'   # change as you like
all_epochs = 200
start_epoch = 1
decay_epoch = 100
lr = 0.0002
in_channel = 3
out_channel = 3
batchSize = 1
size = 256
n_cpu = 2
use_cuda = True  # remember using GPU

G_A2B = Generator(in_channel, out_channel)
G_B2A = Generator(out_channel, in_channel)
D_A = Discriminator(in_channel)
D_B = Discriminator(out_channel)

device = torch.device("cuda")

G_A2B.to(device)
G_B2A.to(device)
D_A.to(device)
D_B.to(device)

G_A2B.apply(init_weight)
G_B2A.apply(init_weight)
D_A.apply(init_weight)
D_B.apply(init_weight)

criterionGAN = nn.MSELoss()
criterionCycle = nn.L1Loss()
criterionIdt = nn.L1Loss()

optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr, betas=(0.5, 0.999))

def lr_lambda(epoch):
    return 1.0 - max(0, epoch + start_epoch - decay_epoch) / (all_epochs - decay_epoch)

lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda)
lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda)

inA = torch.zeros(batchSize, in_channel, size, size, device=device, dtype=torch.float)
inB = torch.zeros(batchSize, in_channel, size, size, device=device, dtype=torch.float)
target_real = torch.ones(batchSize, 1, device=device, dtype=torch.float)
target_fake = torch.zeros(batchSize, 1, device=device, dtype=torch.float)

fake_A_buffer = ImageBuffer()
fake_B_buffer = ImageBuffer()

dataloader = DataLoader(ImageDataset(root, size, 'train'),batchSize,num_workers=n_cpu,drop_last=True)

###      cell4 END      ###



###      cell5 BEGIN     ###

# At first, RESUME=False
# If have trained for several epochs, start_epoch and RESUME need to change

start_epoch = 1
logger = LogInfo(start_epoch, all_epochs, len(dataloader))

RESUME = False
if RESUME:
    device = torch.device("cuda")
    path_checkpoint = 'drive/MyDrive/MyCycleGAN/paras_colab/checkpoint_%s.pth' % str(start_epoch-1)
    new_checkpoint = torch.load(path_checkpoint)
    G_A2B.load_state_dict(new_checkpoint['GA2B'])
    G_B2A.load_state_dict(new_checkpoint['GB2A'])
    D_A.load_state_dict(new_checkpoint['DA'])
    D_B.load_state_dict(new_checkpoint['DB'])
    optimizer_G.load_state_dict(new_checkpoint['OPG'])
    optimizer_D_A.load_state_dict(new_checkpoint['OPDA'])
    optimizer_D_B.load_state_dict(new_checkpoint['OPDB'])
    lr_scheduler_G.load_state_dict(new_checkpoint['LRG'])
    lr_scheduler_D_A.load_state_dict(new_checkpoint['LRDA'])
    lr_scheduler_D_B.load_state_dict(new_checkpoint['LRDB'])
    G_A2B.to(device)
    G_B2A.to(device)
    D_A.to(device)
    D_B.to(device)

for now_epoch in range(start_epoch, all_epochs):

    for i, batch in enumerate(dataloader):
        if (i % 5 == 0):
            print("\nbatch %d" % i)
        # Train input: 2 image from trainA and trainB
        real_A = inA.copy_(batch['A'])
        real_B = inB.copy_(batch['B'])
        """
        Generator 
        """
        optimizer_G.zero_grad()
        # Identity loss
        loss_idt_A = criterionIdt(G_B2A(real_A), real_A) * 5.0
        loss_idt_B = criterionIdt(G_A2B(real_B), real_B) * 5.0
        # GAN loss
        fake_A = G_B2A(real_B)
        fake_B = G_A2B(real_A)
        loss_GAN_A2B = criterionGAN(D_B(fake_B), target_real)
        loss_GAN_B2A = criterionGAN(D_A(fake_A), target_real)
        # Cycle loss
        loss_cycle_ABA = criterionCycle(G_B2A(fake_B), real_A) * 10.0
        loss_cycle_BAB = criterionCycle(G_A2B(fake_A), real_B) * 10.0
        # Total loss
        loss_G_total = loss_idt_A + loss_idt_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        loss_G_total.backward()
        optimizer_G.step()
        """
        Discriminator A
        """
        optimizer_D_A.zero_grad()

        # Real loss
        loss_DA_real = criterionGAN(D_A(real_A), target_real) * 0.5
        # Fake loss
        fake_A = fake_A_buffer.query(fake_A)
        loss_DA_fake = criterionGAN(D_A(fake_A.detach()), target_fake) * 0.5
        # Total loss
        loss_DA_total = loss_DA_real + loss_DA_fake

        loss_DA_total.backward()
        optimizer_D_A.step()
        """
        # Discriminator B #
        """
        optimizer_D_B.zero_grad()
        # Real loss
        loss_DB_real = criterionGAN(D_B(real_B), target_real) * 0.5
        # Fake loss
        fake_B = fake_B_buffer.query(fake_B)
        loss_DB_fake = criterionGAN(D_B(fake_B.detach()), target_fake) * 0.5
        # Total loss
        loss_DB_total = loss_DB_real + loss_DB_fake

        loss_DB_total.backward()
        optimizer_D_B.step()

        # log
        logger.log({'loss_G': loss_G_total, 'loss_G_identity': (loss_idt_A + loss_idt_B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_DA_total + loss_DB_total)},
                   images={'fake_A': fake_A, 'fake_B': fake_B})

    # update lr
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if (now_epoch % 3 == 0):
      # save models checkpoints
      checkpoint = {
          'GA2B': G_A2B.state_dict(),
          'GB2A': G_B2A.state_dict(),
          'DA': D_A.state_dict(),
          'DB': D_B.state_dict(),
          'OPG': optimizer_G.state_dict(),
          'OPDA': optimizer_D_A.state_dict(),
          'OPDB': optimizer_D_B.state_dict(),
          'LRG': lr_scheduler_G.state_dict(),
          'LRDA': lr_scheduler_D_A.state_dict(),
          'LRDB': lr_scheduler_D_B.state_dict(),
          'NOW_EPOCH': now_epoch
      }
      checkpoint_path = 'drive/MyDrive/MyCycleGAN/paras_colab/checkpoint_%s.pth' % (str(now_epoch))
      torch.save(checkpoint, checkpoint_path)
    ckpt={
       'GA2B': G_A2B.state_dict(),
       'GB2A': G_B2A.state_dict()
    }
    checkpoint_path = 'drive/MyDrive/MyCycleGAN/checkpoints_colab/checkpoint_%s.pth' % (str(now_epoch))
    torch.save(ckpt,checkpoint_path)

###      cell5 END      ###