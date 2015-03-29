#!/usr/bin/env python
import os
import argparse
import json
from skimage import io, util
from mpi4py import MPI
from training_experiment import TrainingExperiment

# def get_image_paths(noisy_image_dir, clear_image_dir):
#     # Enumerate the files
#     paths = []
#     # for noisy_image_file in sorted(os.listdir(noisy_image_dir))[0:4]:
#     for noisy_image_file in sorted(os.listdir(noisy_image_dir)):
#         name, ext = os.path.splitext(noisy_image_file)
#         contrast = name.split('-')[::-1][0]
#         clear_image_name = 'clear-00-00-' + contrast + ext
#         noisy_image_path = os.path.join(noisy_image_dir, noisy_image_file)
#         clear_image_path = os.path.join(clear_image_dir, clear_image_name)
#         paths.append((noisy_image_path, clear_image_path))
#     return paths


def get_images(noisy_image_dir, clear_image_dir):
    images = []
    for noisy_image_file in sorted(os.listdir(noisy_image_dir)):
        name, ext = os.path.splitext(noisy_image_file)
        contrast = name.split('-')[::-1][0]
        clear_image_name = 'clear-00-00-' + contrast + ext
        noisy_image_path = os.path.join(noisy_image_dir, noisy_image_file)
        clear_image_path = os.path.join(clear_image_dir, clear_image_name)
        # Read images
        noisy_image = util.img_as_float(io.imread(noisy_image_path))
        clear_image = util.img_as_float(io.imread(clear_image_path))
        images.append((noisy_image, clear_image, name))
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'noisy_image_dir', help='Noisy image directory')
    parser.add_argument(
        'clear_image_dir', help='Clear image directory')
    parser.add_argument(
        'result_dir', help='Result directory')
    parser.add_argument(
        'params', help='Param file (json)')
    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    proc_id = comm.Get_rank()
    proc_count = comm.Get_size()

    if proc_id == 0:
        # Distribute images
        images = get_images(
            args.noisy_image_dir,
            args.clear_image_dir)
        # Distribute params
        with open(args.params, 'r') as fp:
            params = json.load(fp)
    else:
        images = None
        params = None

    images = comm.scatter(images, root=0)
    params = comm.bcast(params, root=0)
    noisy_image, clear_image, name = images
    # print proc_id, name, str(params)
    
    # Separate result directory for each image
    image_result_dir = os.path.join(
        args.result_dir, name)
    if not os.path.exists(image_result_dir):
        os.makedirs(image_result_dir)

    # Start the experiment
    exp = TrainingExperiment(
        noisy_image, clear_image, image_result_dir, params)
    exp.run()
