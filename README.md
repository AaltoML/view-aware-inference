# Gaussian Process Priors for View-Aware Inference

Code for the paper:
* Hou, Y., Heljakka, A., Kannala, J., and Solin, A. (2020). **Gaussian Process Priors for View-Aware Inference**. In *Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21), to appear*. [[arXiv preprint]](https://arxiv.org/abs/1912.03249)

Implementation by **Ari Heljakka** and **Yuxin Hou**. (StyleGAN, perceptual similarity metrics and StyleGAN projection code adapted from [1-3]. GPPVAE code adapted from [4])

## Data

For view synthesis with a GP Prior VAE, please download the data from [Google Drive](https://drive.google.com/file/d/19WPAU8VcPJ5UgLz_qdi_n97odMx-L21s/view?usp=sharing). The dataset contains a h5df file(data_chairs.h5) and a numpy array that store the camera poses for 60 different viewpoints for training(pose60.npy), and 60 novel viewpoints for evaluation(seq_pose.npy)

For face reconstruction, please download the dataset from [Google Drive](https://drive.google.com/drive/folders/1lc2YVHq_ZYHsbHRorf9K_KBSTT-fG_Kg).
The dataset contains the four full camera runs, created by only keeping every 5th frame.
These PNGs have already been aligned and cropped to 512x512 so as to be suitable for StyleGAN projection.
The StyleGAN latent codes are already included, so you can skip the non-deterministic projection step. Still, should you wish to do so for your new images, please run `python project_z.py  --img_input_path [your data]`.

## System Requirements

A GPU with 11 GB of RAM is required for StyleGAN generation (and optional projection) steps.

For all dependencies of face reconstruction experiments, please run 
```
conda install --file requirements. txt
```

For view synthesis with a GP Prior VAE experiments, simply install `h5py`.

## Training the VAE models
First, we train the basic VAE model
```
python ./gppvae/train_vae.py --data data_chairs.h5 --outdir ./out/vae
```
Then, we train the GPPVAE
```
python ./gppvae/train_gppvae.py --data data_chairs.h5 --pose pose60.npy --outdir ./out/gppvae --vae_cfg ./out/vae/vae.cfg.p --vae_weights ./out/vae/weights/weights.00990.pt
```

## Reproducing the Paper Results for View Synthesis with a GP Prior VAE

First, please download our pre-trained model from [Google Drive](https://drive.google.com/file/d/1DVg0CT1WlhipxflPGjwj_8jy0r6MxPCG/view?usp=sharing)

Then, to test the performance of the original task, run
```
python test_vae.py --data data_chairs.h5 --pose pose60.npy --vae_cfg out/vae/vae.cfg.p \
--vae_weights ./out/gppvae/weights/vae_weights.00110.pt \
--gp_weights ./out/gppvae/weights/gp_weights.00110.pt \
--vm_weights ./out/gppvae/weights/vm_weights.00110.pt 
```

To predict novel views that are not presented in the training set, please check the notebook file `./gppvae/novel_view_prediction.ipynb`

## Reproducing the Paper Results for Face Reconstruction

For the face reconstruction results, please run 
```
./evaluate.sh [data path]
```

This command runs the following steps:
1. Load frames.csv, convert it into distance matrices for GP operations, and execute all interpolation modes, including the baselines, to produce the corrected latent Z matrices usable for StyleGAN.
2. Decode each Z array with StyleGAN to X image frames.
3. For quantitative metrics of images, crop every image in order to run LPIPS on the X frames, both between methods and within the consequtive frames produced by each method.
4. Compute the statistics into lpips_... files.
5. Gather and print out the statistics.

You can run step #1 manually for e.g. face id #2 with (first, last) interpolation with
```
python eval.py --data_path ./data  --face_id=2
```

For the baseline separable (Euler) kernels and the quaternion kernels, run with `--kernel_mode=[quat|euler]`.
For the smoothing with all frames, run with `--full_smoothing`.

For other steps, please see `evaluate.sh` for example commands.

# References

[1] Karras, T., Laine, S., Aila, T. (2019). **A Style-Based Generator Architecture for Generative Adversarial Networks**. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. https://github.com/tkarras/progressive_growing_of_gans

[2] Zhang, R. and Isola, P. and Efros, A. A. and Shechtman, E. and Wang, O. (2018). **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric** In: *CVPR*. https://github.com/richzhang/PerceptualSimilarity

[3] https://github.com/Puzer/stylegan-encoder

[4] Casale, F. P., Dalca, A. V., Saglietti, L., Listgarten, J., & Fusi, N. (2018). **Gaussian process prior variational autoencoders** In *NeurIPS*.   https://github.com/fpcasale/GPPVAE

[5] https://shapenet.org/
