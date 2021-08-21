import random
import time
import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    # 1 channel to 3 channels
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


class ImageBuffer:
    """
        receive the latest images from Generator and store them
        return input images with 0.5 probability
        return previous generated images with 0.5 probability
    """

    def __init__(self, max_size=50):
        self.max_size = max_size
        self.num_image = 0
        self.data = []

    def query(self, input_images):
        if self.max_size == 0:
            return input_images
        to_return = []
        for image in input_images:
            image = torch.unsqueeze(image.detach(), 0)
            if self.num_image < self.max_size:  # Buffer is not full
                self.num_image += 1
                self.data.append(image)
                to_return.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    index = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[index].clone())
                    self.data[index] = image
                else:
                    to_return.append(image)
        return torch.cat(to_return)


class LogInfo:
    def __init__(self, start_epoch, all_epochs, batches_per_epoch):
        self.start_epoch = start_epoch
        self.all_epochs = all_epochs
        self.writer = SummaryWriter('logs')
        self.losses = {}
        self.images = {}
        self.batches_per_epoch = batches_per_epoch
        self.now_epoch = start_epoch
        self.now_batch = 1
        self.time_used = 0
        self.prev_time = time.time()
        self.n_draw = 1

    # need 2 dicts
    def log(self, losses=None, images=None):
        now_time = time.time()
        self.time_used += now_time - self.prev_time
        self.prev_time = now_time
        print('\r-- Epoch: %03d/%03d [%04d/%04d] --' % (
            self.now_epoch, self.all_epochs, self.now_batch, self.batches_per_epoch), end=" ")
        # accumulate different type of losses and print
        for index, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (index + 1) == len(losses.keys()):
                # accumulate and divide to get average loss
                print('%s: %.4f --' % (loss_name, self.losses[loss_name] / self.now_batch), end=" ")
            else:
                print('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.now_batch), end=" ")

        batches_done = self.batches_per_epoch * (self.now_epoch - 1) + self.now_batch
        batches_left = self.batches_per_epoch * self.all_epochs - batches_done
        # Estimated Time of Arrival
        print(
            "Estimated Time of Arrival: %s" % datetime.timedelta(seconds=batches_left * self.time_used / batches_done))
        # Draw Image
        for image_name, image_tensor in images.items():  # .items() return (key,value)
            self.writer.add_image(image_name, tensor2image(image_tensor.detach()), global_step=self.n_draw)
        self.n_draw += 1

        # End of one epoch
        if (self.now_batch % self.batches_per_epoch) == 0:
            # Plot losses using tensorboard
            for loss_name, loss in self.losses.items():
                self.writer.add_scalar(loss_name, scalar_value=np.array([loss / self.now_batch]),
                                       global_step=self.now_epoch)
                self.losses[loss_name] = 0.0
            # update
            self.now_batch = 1
            self.now_epoch += 1
            print('\n')
        else:
            self.now_batch += 1

    def endplot(self):
        self.writer.close()
