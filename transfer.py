import argparse
import os
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
import torch
from network import Generator
from datasets import ImageDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model_root', type=str, default='trained_model/white2yellow')
    opt = parser.parse_args()

    # Variable

    in_channel = out_channel = 3
    use_cuda = opt.cuda
    model_root = opt.model_root + '/model.pth'
    batchSize = 1
    size = 256
    dataroot = 'apply/'
    n_cpu = 2

    G_A2B = Generator(in_channel, out_channel)
    G_B2A = Generator(out_channel, in_channel)

    checkpoint = torch.load(model_root)

    device = torch.device("cuda" if use_cuda else "cpu")

    G_A2B.to(device)
    G_B2A.to(device)

    G_A2B.load_state_dict(checkpoint['GA2B'])
    G_B2A.load_state_dict(checkpoint['GB2A'])

    G_A2B.eval()
    G_B2A.eval()

    # input
    inA = torch.zeros(batchSize, in_channel, size, size, device=device, dtype=torch.float)
    inB = torch.zeros(batchSize, in_channel, size, size, device=device, dtype=torch.float)

    new_ImageDataset = ImageDataset(dataroot, size, mode='in')

    # dataloader
    dataloader = DataLoader(new_ImageDataset, batchSize, shuffle=False,
                            num_workers=n_cpu)

    A_isNotEmpty = new_ImageDataset.A_isNotEmpty()
    B_isNotEmpty = new_ImageDataset.B_isNotEmpty()

    # Create folder to store output
    if not os.path.exists('apply/outputA'):
        os.makedirs('apply/outputA')
    if not os.path.exists('apply/outputB'):
        os.makedirs('apply/outputB')

    for index, batch in enumerate(dataloader):

        if A_isNotEmpty:
            real_A = inA.copy_((batch['A'])[0])
            fake_B = 0.5 * (G_A2B(real_A) + 1.0)
            save_image(fake_B, 'apply/outputA/%s' % (batch['A'])[1])
        if B_isNotEmpty:
            real_B = inB.copy_((batch['B'])[0])
            fake_A = 0.5 * (G_B2A(real_B) + 1.0)
            save_image(fake_A, 'apply/outputB/%s' % (batch['B'])[1])

        print('Success: %04d of %04d' % (index + 1, len(dataloader)))

    print('Done!')


