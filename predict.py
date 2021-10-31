import os
import tempfile
import shutil
from pathlib import Path
import argparse
from PIL import Image
import cog
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util


class Predictor(cog.Predictor):
    def setup(self):
        parser = argparse.ArgumentParser()
        TestOptions().initialize(parser)
        self.opt = parser.parse_args(
            ["--content_path", "", "--style_path", "", "--name", "AdaAttN", "--model", "adaattn", "--dataset_mode",
             "unaligned", "--load_size", "512", "--crop_size", "512"])
        self.opt.isTrain = False
        self.opt.gpu_ids = [0]
        self.opt.image_encoder_path = 'checkpoints/vgg_normalised.pth'
        self.opt.skip_connection_3 = True
        self.opt.shallow_layer = True
        self.opt.num_threads = 0  # test code only supports num_threads = 0
        self.opt.batch_size = 1  # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

        self.model = create_model(self.opt)  # create a model given opt.model and other options
        self.model.setup(self.opt)
        # self.device = torch.device('cuda:0')

    @cog.input(
        "content",
        type=Path,
        help="input content image"
    )
    @cog.input(
        "style",
        type=Path,
        help="input style image"
    )
    def predict(self, content, style):
        input_content_dir = 'temp_content'
        input_style_dir = 'temp_style'
        os.makedirs(input_content_dir, exist_ok=True)
        os.makedirs(input_style_dir, exist_ok=True)
        try:
            content_path = os.path.join(input_content_dir, os.path.basename(content))
            style_path = os.path.join(input_style_dir, os.path.basename(style))
            shutil.copy(str(content), content_path)
            shutil.copy(str(style), style_path)
            self.opt.content_path = input_content_dir
            self.opt.style_path = input_style_dir
            dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options

            for data in dataset:
                self.model.set_input(data)  # unpack data from data loader
                self.model.test()  # run inference
                visuals = self.model.get_current_visuals()  # get image results
                out_path = Path(tempfile.mkdtemp()) / "out.png"
                im = util.tensor2im(visuals['cs'])
                image_pil = Image.fromarray(im)
                image_pil.save(str(out_path))
        finally:
            clean_folder(input_content_dir)
            clean_folder(input_style_dir)

        return out_path


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
