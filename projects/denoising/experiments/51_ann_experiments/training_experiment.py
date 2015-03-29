import os
import sys
import select
import tempfile
import json
import time
from fann2 import libfann
from projects.denoising.neural.filtering import get_training_data
# from skimage import io, util
import skimage.io
import skimage.util
from cStringIO import StringIO

ACTIVATION_FUNCTIONS = {
    'sigmoid': libfann.SIGMOID,
    'gaussian': libfann.GAUSSIAN,
    'elliot': libfann.ELLIOT
}


# def create_training_data(
#         noisy_image_dir, clear_image_dir,
#         patch_size, output_dir):
#     """
#     Creates FANN-ready training data for each noisy-clear image pair
#     and writes it into specified
#     """
#     for noisy_image_file in sorted(os.listdir(noisy_image_dir))[0:4]:
#         name, ext = os.path.splitext(noisy_image_file)
#         contrast = name.split('-')[::-1][0]
#         clear_image_name = 'clear-00-00-' + contrast + ext
#         noisy_image_path = os.path.join(noisy_image_dir, noisy_image_file)
#         clear_image_path = os.path.join(clear_image_dir, clear_image_name)

#         # Open the images
#         noisy_image = skimage.util.img_as_float(
#             skimage.io.imread(noisy_image_path))
#         clear_image = skimage.util.img_as_float(
#             skimage.io.imread(clear_image_path))

#         # Make training data
#         inputs, outputs = get_training_data(
#             noisy_image,
#             clear_image,
#             patch_size=patch_size)

#         # Create .data file
#         result_filename = name + '.data'
#         result_path = os.path.join(output_dir, result_filename)
#         with open(result_path, 'w') as fp:
#             header = "%i %i %i\n" % (
#                 len(inputs), len(inputs[0]), 1)
#             fp.write(header)

#             content = "\n".join([
#                 ' '.join(["{:1.6f}".format(num) for num in input]) + '\n' + "{:1.6f}".format(output)
#                 for input, output in zip(inputs, outputs)
#             ])
#             fp.write(content)
#     return None


# http://stackoverflow.com/questions/9488560/capturing-print-output-from-shared-library-called-from-python-with-ctypes-module
class StdoutPipeCapture(object):
    def __enter__(self):
        # the pipe would fail for some reason if I didn't write to stdout at some point
        # so I write a space, then backspace (will show as empty in a normal terminal)
        sys.stdout.write(' \b')
        self.pipe_out, self.pipe_in = os.pipe()
        # save a copy of stdout
        self.stdout = os.dup(1)
        # replace stdout with our write pipe
        os.dup2(self.pipe_in, 1)
        return self

    # check if we have more to read from the pipe
    def more_data(self):
        r, _, _ = select.select([self.pipe_out], [], [], 0)
        return bool(r)

    # read the whole pipe
    def read_pipe(self):
        out = ''
        while self.more_data():
            out += os.read(self.pipe_out, 1024)
        return out

    def __exit__(self, *args):
        # put stdout back in place
        os.dup2(self.stdout, 1)
        self.output = self.read_pipe()


class TrainingExperiment(object):
    def __init__(
            self, noisy_image, clear_image, result_dir, params):
        self.noisy_image = noisy_image
        self.clear_image = clear_image
        self.result_dir = result_dir
        self.params = params
        # self.params = {
        #     'id': 'ABCD-1234',
        #     'patch_size': 3,
        #     'activation_function': 'elliot',
        #     'hidden_neurons': 2,
        # }

    def run(self):
        """
        Train ANN with one noisy and one clear image
        """
        ann = libfann.neural_net()
        topology = [
            self.params['patch_size'] ** 2,
            self.params['hidden_neurons'],
            1
        ]
        ann.create_standard_array(topology)
        ann.set_learning_rate(self.params['learning_rate'])
        # Set AF for whole hidden layer
        ann.set_activation_function_layer(
            ACTIVATION_FUNCTIONS[self.params['activation_function']], 1)

        # Train
        training_file = self._get_training_file()

        # Timer start
        start = time.time()

        # Capture stdout from libfann
        with StdoutPipeCapture() as capture:
            ann.train_on_file(
                training_file.name,
                self.params['max_iterations'],
                self.params['iterations_between_reports'],
                self.params['desired_error']
            )

        # Timer stop
        end = time.time()
        duration = end - start

        # Save trained network
        ann_path = os.path.join(
            self.result_dir, str(self.params['id']) + '.net')
        ann.save(ann_path)
        
        # Save overall results
        results = {}
        results['params'] = self.params
        results['output'] = capture.output
        results['training_time'] = duration

        results_path = os.path.join(
            self.result_dir, str(self.params['id']) + '.json')
        with open(results_path, 'w') as fp:
            json.dump(results, fp)

        # Cleanup
        training_file.close()


    def _get_training_file(self):
        """
        Make training data from noisy+clear images
        and write it into temporary file
        """
        # noisy_image = skimage.util.img_as_float(
        #     skimage.io.imread(
        #         self.noisy_image_path))
        # clear_image = skimage.util.img_as_float(
        #     skimage.io.imread(
        #         self.clear_image_path))
        inputs, outputs = get_training_data(
            self.noisy_image,
            self.clear_image,
            patch_size=self.params['patch_size'])

        temp_file = tempfile.NamedTemporaryFile()
        header = "%i %i %i\n" % (
            len(inputs), len(inputs[0]), 1)
        temp_file.write(header)

        content = "\n".join([
            ' '.join(["{:1.6f}".format(num) for num in input]) + '\n' + "{:1.6f}".format(output)
            for input, output in zip(inputs, outputs)
        ])
        temp_file.write(content)
        temp_file.flush()
        return temp_file
