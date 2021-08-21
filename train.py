import argparse
import itertools
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from network import Generator
from network import Discriminator
from utils import ImageBuffer
from datasets import ImageDataset
from utils import LogInfo

if __name__ == '__main__':

    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(m.bias.data, val=0.0)


    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--all_epochs', type=int, default=200)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--root', type=str, default='datasets/cloudtransfer/')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--size', type=int, default=256)
    # action = 'store_true' : opt.cuda == True if we use --cuda command
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--out_channel', type=int, default=3)

    opt, unknown = parser.parse_known_args()

    # define variables to make debug convenient
    root = opt.root
    all_epochs = opt.all_epochs
    start_epoch = opt.start_epoch
    lr = opt.lr
    in_channel = opt.in_channel
    out_channel = opt.out_channel
    batchSize = opt.batchSize
    size = opt.size
    use_cuda = opt.cuda

    # Network:
    G_A2B = Generator(in_channel, out_channel)
    G_B2A = Generator(out_channel, in_channel)
    D_A = Discriminator(in_channel)
    D_B = Discriminator(out_channel)

    # CUDA ON
    device = torch.device("cuda" if use_cuda else "cpu")
    G_A2B.to(device)
    G_B2A.to(device)
    D_A.to(device)
    D_B.to(device)

    G_A2B.apply(init_weight)
    G_B2A.apply(init_weight)
    D_A.apply(init_weight)
    D_B.apply(init_weight)

    # Loss
    criterionGAN = nn.MSELoss()
    criterionCycle = nn.L1Loss()
    criterionIdt = nn.L1Loss()

    # Optimizer
    optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr, betas=(0.5, 0.999))

    # LR scheduler
    # in epoch_i, new_lr = old_lr * lr_lambda(epoch_i)
    def lr_lambda(epoch):
        return 1.0 - max(0, epoch + opt.start_epoch - opt.decay_epoch) / (opt.all_epochs - opt.decay_epoch)


    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda)

    # input and target
    inA = torch.zeros(batchSize, in_channel, size, size, device=device, dtype=torch.float)
    inB = torch.zeros(batchSize, in_channel, size, size, device=device, dtype=torch.float)
    target_real = torch.ones(batchSize, 1, device=device, dtype=torch.float)
    target_fake = torch.zeros(batchSize, 1, device=device, dtype=torch.float)
    fake_A_buffer = ImageBuffer()
    fake_B_buffer = ImageBuffer()

    # dataloader
    dataloader = DataLoader(ImageDataset(root, size, 'train'), batchSize, num_workers=opt.n_cpu, drop_last=True)

    # prepare log information
    logger = LogInfo(start_epoch, all_epochs, len(dataloader))

    # Training process
    """
    1.Input
    2.Generator
        2.1 Identity loss: G_A2B(B) should be B
        2.2 GAN loss
        2.3 Cycle loss
    3.Discriminator A & Discriminator B
        3.1 real loss
        3.2 fake loss
    """
    for now_epoch in range(start_epoch, all_epochs):
        for i, batch in enumerate(dataloader):
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
                       images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        # update lr
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # save models checkpoints
        torch.save(G_A2B.state_dict(), 'checkpoints/Generator_A2B.pth')
        torch.save(G_B2A.state_dict(), 'checkpoints/Generator_B2A.pth')
        torch.save(D_A.state_dict(), 'checkpoints/Discriminator_A.pth')
        torch.save(D_B.state_dict(), 'checkpoints/Discriminator_B.pth')

    logger.endplot()
