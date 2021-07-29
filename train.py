import os
import sys
import time
import math
import random
import argparse
import numpy as np
from tqdm import tqdm

import utils
from models.lifted_gan import LiftedGAN

import torch
import torchvision
from tensorboardX import SummaryWriter


def test_batch(model, code, batch_size, keys=None):
    results = {}
    for head in range(0,len(code),batch_size):
        tail = min(len(code), head+batch_size)
        inputs_batch = code[head:tail]
        results_batch = model.test(inputs_batch, render=True, recon_normal=False, generate=False)

        if keys is None: keys = results_batch.keys()
        for k in keys:
            results[k] = results.get(k, list()) + [results_batch[k]]

    for k in results.keys():
        results[k] = torch.cat(results[k],0)
    return results

        
def main(args):
    # I/O
    config_file = args.config_file
    config = utils.import_file(config_file, 'config_')

    if config.base_random_seed is not None:
        random.seed(config.base_random_seed)
        torch.manual_seed(config.base_random_seed)

    network = LiftedGAN()
    network.initialize(config)

    # Initalization for running
    log_dir = utils.create_log_dir(config.log_base_dir, config.name, config_file)
    summary_writer = SummaryWriter(log_dir)
    if config.restore_model:
        start_epoch = network.restore_model(config.restore_model)
    else:
        start_epoch = 0

    # Initialize the random codes for sampling
    sample_codes = torch.rand(config.n_samples, 512)

    # Main Loop
    print(f'\nStart Training\n# epochs: {config.num_epochs}\nbatch_size: {config.batch_size}\n')
    for epoch in range(start_epoch, config.num_epochs):

        # Training
        start_time = time.time()
        for step in range(config.epoch_size):

            watchlist, summary, global_step = network.train_step()

            # Display
            if step % config.print_interval == 0:
                watchlist['time'] = time.time() - start_time
                utils.display_info(epoch, step, watchlist)
                start_time = time.time()

            if ((global_step-1) % config.summary_interval == 0) or (step==0 and epoch==start_epoch):
                summary = network.add_videos(summary)
                utils.write_summary(summary_writer, summary, global_step)

        # Testing
        results = test_batch(network, sample_codes, batch_size=config.batch_size)
        images = utils.stack_images(results['recon_im'], results['canon_im'])
        torchvision.utils.save_image(images, f'{log_dir}/samples/{global_step}.jpg', nrow=8, normalize=True)
        network.save_model(log_dir, epoch+1)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    args = parser.parse_args()
    main(args)
