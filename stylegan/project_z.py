import argparse
import numpy as np
import tensorflow as tf
import pickle
import PIL.Image
import sys

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
from dnnlib import EasyDict

from face_align import FaceAligner

REF_IMG_SIZE = 1024

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  

config = EasyDict(
    cache_dir='_cache',
    aligned_image='test.png',
    optimizer=dnnlib.EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8),
    learning_rate=0.01,
    num_iters=200
)

def parse_args():
    parser = argparse.ArgumentParser(description='PIONEER')
    parser.add_argument('--z_input_path', type=str, default=None)
    parser.add_argument('--z_input_file', type=str, default=None)
    parser.add_argument('--img_input_path', type=str)
    parser.add_argument('--img_out_path', type=str)
    parser.add_argument('--mode', type=str, help='reco|decode')
    parser.add_argument('--load_start_index', type=int, default=-1)
    parser.add_argument('--z_interpolate_range', type=str, help='[start__index,diff,end_index]')

    return parser.parse_args()

args = parse_args()

def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir=config.cache_dir)
    return open(file_or_url, 'rb')

def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding='latin1')

def from_uint8(x):
    return tf.cast(x, tf.float32) / 255.0 - 0.5

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

class LatentProjector():
    def __init__(self, aligner, Gs=None, distance_measure=None):
        self.learning_rate = config.learning_rate
        self.aligner = aligner
        
        if distance_measure is None:
            distance_measure = load_pkl('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2') # vgg16_zhang_perceptual.pkl
        self.distance_measure = distance_measure

        if Gs is None:
            print ('Loading stylegan pickle')
            with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
                generator_network, discriminator_network, Gs = pickle.load(f)
        self.Gs = Gs

        with tf.name_scope('Inputs'):
            with tf.device('/cpu:0'):
                self.lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
                self.dlatent_in = tf.placeholder(tf.float32, name='dlatent_in', shape=[1,18,512])

            with tf.device('/gpu:0'):
                self.target_image_in = tf.placeholder(tf.uint8, name='target_image_in', shape=[1,3,REF_IMG_SIZE,REF_IMG_SIZE])

        dlatent_avg = np.tile(Gs.get_var('dlatent_avg'), (18,1))
        self.dlatent_var = tf.Variable(dlatent_avg[np.newaxis])
        assert self.dlatent_var.get_shape() == (1,18,512)

        self.gen_img = Gs.components.synthesis.get_output_for(
            self.dlatent_var,
            randomize_noise=True,
            structure='fixed'
        )

        mask = np.zeros(shape=self.target_image_in.shape.as_list(), dtype=np.float32)
        mask[:, :, 64:192, 64:192] = 1.0
        img1 = tf.cast(self.target_image_in, tf.float32)
        img2 = nhwc_to_nchw(tf.image.resize_images(nchw_to_nhwc((self.gen_img + 1) * 127.5), size=(REF_IMG_SIZE, REF_IMG_SIZE), method=2))
        self.loss_val = self.distance_measure.get_output_for(
            img1*(1.0-mask) + 0.5*mask,
            img2*(1.0-mask) + 0.5*mask
        )

        opt = tflib.Optimizer(learning_rate=self.lrate_in, **config.optimizer)
        with tf.control_dependencies([autosummary("Loss", self.loss_val)]):
            opt.register_gradients(self.loss_val, [self.dlatent_var])
        self.train_step = opt.apply_updates()
        tflib.run([self.dlatent_var.initializer])

        # Reset op
        self.dlatent_reset_op = tf.assign(self.dlatent_var, self.dlatent_in)

    def run_train_step(self, target_image=None, learning_rate=None):
        img = np.array(target_image, dtype=np.uint8).transpose([2,0,1])[np.newaxis]
        #import ipdb; ipdb.set_trace()

        tflib.run([self.train_step], {self.lrate_in: learning_rate, self.target_image_in: img})
        return self.dlatent_var.eval()

    def synth_image(self, savepath, loadpath=None, zzz=None):
        # Dump output image
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        print(np.shape(self.dlatent_var.eval()))
        if loadpath is None and zzz is None:
            zzz = self.dlatent_var.eval()
            np.save(savepath, zzz)
        elif zzz is None:
            zzz = np.load(loadpath)
            print("Loaded z from path {}".format(loadpath))
            print(np.shape(zzz))

        result_imgs = self.Gs.components.synthesis.run(
            zzz,
            randomize_noise=False,
            structure='fixed',
            **synthesis_kwargs
        )
        return result_imgs[0]

    def reset_dlatent(self):
        dlatent_avg = np.tile(self.Gs.get_var('dlatent_avg'), (18,1))[np.newaxis]
        tflib.run([self.dlatent_reset_op], {self.dlatent_in: dlatent_avg})
    def stest_init(self):
        rnd = np.random.RandomState(5)
        self.latents = rnd.randn(1, self.Gs.input_shape[1])
    def stest(self):
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = self.Gs.run(self.latents, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)
        return images

def main():
    global args

    if tf.get_default_session() is None:
        print('Initializing TensorFlow...')
        tflib.init_tf()

    input_path = args.img_input_path
    prefix = ''
    ext = 'png'
    first = int(args.load_start_index)
    do_align = (first != -1)

    assert(args.z_input_path is None or args.z_input_file is None)

    if not do_align:
        projector = LatentProjector(Gs=None, aligner=None)

        if not args.z_input_file is None:
            print("Load images from z at {}".format(args.z_input_file))

            z_all = np.load(args.z_input_file)
            for i, z_i in enumerate(z_all):
                if i%5 == 0:
                    gend_image = projector.synth_image(None, zzz=z_i.reshape([1,18,512])) 
                    outname = ('{}/{}.{}'.format(args.img_out_path,format(i+1, '04d'), ext))
                    PIL.Image.fromarray(gend_image).save(outname)
                    print("to: {}".format(outname))
        elif not args.z_interpolate_range is None:
            print("Interpolate linearly from directory containing separate z vector files at {}".format(args.z_input_path))


            z_start, z_delta_N, z_end = tuple([int(ii) for ii in args.z_interpolate_range.split(',')])

            print((z_start, z_delta_N, z_end))

            p_end   = args.z_input_path +  '/{}.z0.npy'.format(format(z_end,'04d'))
            p_start = args.z_input_path +  '/{}.z0.npy'.format(format(z_start+1,'04d'))
            z_end_vector = np.load(p_end)
            z_start_vector = np.load(p_start)
            z_parts = (z_end - z_start) / (z_delta_N+1)
            z_delta = (z_end_vector - z_start_vector) / z_parts

            print('Total frames incl. start and end is {}'.format(z_parts))
            for i in range(int((z_end-z_start+1)/z_delta_N+1)):
                z_i = z_start_vector + i*z_delta
                gend_image = projector.synth_image(None, zzz=z_i.reshape([1,18,512]))
                outname = ('{}/{}.png'.format(args.img_out_path,format(z_start+i*z_delta_N+1, '04d')))
                PIL.Image.fromarray(gend_image).save(outname)
                print("to: {}".format(outname))
    else:
        img_cycle = 5
        align_only = False
        remaining_gas = 10 #On test workstation, this crashes after 10 with OOM.
        aligner = FaceAligner(config.cache_dir)
        
        for img_j in range(first,1080):
            if img_cycle != 1 and img_j%img_cycle != 1: #1,6,11, ...
                continue
            if remaining_gas <=0:
                break
            remaining_gas -= 1
            img_i = format(img_j, '04d')
            print("Load and align...")
            print('{}/raw/{}{}.{}'.format(input_path, prefix, img_i, ext))
            try:
                tgt = aligner.load_and_align_face('{}/raw/{}{}.{}'.format(input_path, prefix, img_i, ext))
                tgt.save('{}/align/{}{}.{}'.format(input_path, prefix, img_i, ext))
            except:
                print('fail')
                continue

            if align_only:
                continue

            projector = LatentProjector(Gs=None, aligner=aligner)

            print("Iterate...")

            for i in range(config.num_iters):
                projector.run_train_step(target_image=tgt, learning_rate=config.learning_rate)
                if i % 10 == 0:
                    loss = tflib.run(projector.loss_val, {projector.target_image_in: np.reshape(tgt,(1,3,REF_IMG_SIZE,REF_IMG_SIZE)) })
                    print ('Iter', i, 'loss', loss)

            gend_image = projector.synth_image('{}/stylegan/{}{}.z0'.format(input_path,prefix,img_i))
            PIL.Image.fromarray(gend_image).save('{}/stylegan/{}{}.{}'.format(input_path,prefix,img_i,ext)) 

if __name__ == "__main__":
    main()
