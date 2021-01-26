# Gaussian Process Priors for View-Aware Inference

Code for the paper:
* Yuxin Hou, Ari Heljakka, and Arno Solin (2021). **Gaussian Process Priors for View-Aware Inference**. In *Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21), to appear*. [[arXiv preprint]](https://arxiv.org/abs/1912.03249)

Implementation by **Ari Heljakka**. (StyleGAN, perceptual similarity metrics and StyleGAN projection code adapted from [1-3].)

The published implementation covers Sec. 4.2 experiments.
The Sec. 4.1 experiments code and data will be added by March 2021.

# Data

Please download the dataset from [Google Drive](https://drive.google.com/drive/folders/1lc2YVHq_ZYHsbHRorf9K_KBSTT-fG_Kg).
The dataset contains the four full camera runs, created by only keeping every 5th frame.
These PNGs have already been aligned and cropped to 512x512 so as to be suitable for StyleGAN projection.
The StyleGAN latent codes are already included, so you can skip the non-deterministic projection step. Still, should you wish to do so for your new images, please run `python project_z.py  --img_input_path [your data]`.

# System Requirements

A GPU with 11 GB of RAM is required for StyleGAN generation (and optional projection) steps.

For all dependencies, please run 
```
conda install --file requirements. txt
```

# Reproducing the Paper Results

To reproduce the primary results, please run 
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

