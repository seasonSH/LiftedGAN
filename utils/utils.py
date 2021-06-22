import os
import sys
import numpy as np
import imp
import time
from datetime import datetime
import shutil
import torch

def import_file(full_path_to_module, name='module.name'):

    module_obj = imp.load_source(name, full_path_to_module)

    return module_obj

def create_log_dir(log_base_dir, name, config_file):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(log_base_dir), name, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    shutil.copyfile(config_file, os.path.join(log_dir,'config.py'))
    os.makedirs(os.path.join(log_dir,'samples'))

    return log_dir


def display_info(epoch, step, watch_list):
    sys.stdout.write('[%d][%d]' % (epoch+1, step+1))
    for item in watch_list.items():
        if type(item[1]) in [float, np.float32, np.float64]:
            sys.stdout.write('   %s: %2.3f' % (item[0], item[1]))
        elif type(item[1]) in [int, bool, np.int32, np.int64, np.bool]:
            sys.stdout.write('   %s: %d' % (item[0], item[1]))
    sys.stdout.write('\n')


def write_summary(summary_writer, summary, global_step):
    for k,v in summary['scalar'].items():
        summary_writer.add_scalar(k, v, global_step)
    for k,v in summary['histogram'].items():
        summary_writer.add_histogram(k, v, global_step)
    for k,v in summary['image'].items():
        summary_writer.add_image(k, v, global_step)
    for k,v in summary['video'].items():
        summary_writer.add_video(k, v, global_step, fps=4)
    summary_writer.file_writer.flush()


def stack_images(*images):
    num_images = len(images)
    with torch.no_grad():
        n,c,h,w = images[0].size()
        images_stack = torch.stack(images, 1)
        images_stack = images_stack.reshape(num_images*n,c,h,w)
        return images_stack
