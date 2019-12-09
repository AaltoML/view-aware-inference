## Gaussian Process Priors for View-Aware Inference

Project page for:

* Yuxin Hou, Ari Heljakka, and Arno Solin (2019). **Gaussian Process Priors for View-Aware Inference**. [[arXiv preprint]]( http://arxiv.org/abs/1912.03249)


### Abstract

We derive a principled framework for encoding prior knowledge of information coupling between views or camera poses (translation and orientation) of a single scene. While deep neural networks have become the prominent solution to many tasks in computer vision, some important problems not so well suited for deep models have received less attention. These include uncertainty quantification, auxiliary data fusion, and real-time processing, which are instrumental for delivering practical methods with robust inference. While these are central goals in probabilistic machine learning, there is a tangible gap between the theory and practice of applying probabilistic methods to many modern vision problems. For this, we derive a novel parametric kernel (covariance function) in the pose space, SE(3), that encodes information about input pose relationships into larger models. We show how this soft-prior knowledge can be applied to improve performance on several real vision tasks, such as feature tracking, human face encoding, and view synthesis.

![](assets/fig/view-aware.jpg)

*Illustrative sketch of the logic of the proposed method: We propose a Gaussian process prior for encoding known six degrees-of-freedom camera movement (relative pose information) into probabilistic models. In this example, built-in visual- inertial tracking of the iPhone movement is used for pose estimation. The phone starts from standstill at the left and moves to the right (translation can be seen in the covariance in (a)). The phone then rotates from portrait to landscape which can be read from the orientation (view) covariance in (b).*


### Example: View synthesis with the view-aware GP prior

The paper features several examples of the applicability of the proposed model. The last example, however, is best demonstrated with videos. Thus we have included a set of videos below for complementing the result figures in the paper.

We consider the task of face reconstruction using a GAN model. As input we use a video captured with an Apple iPhone (where we also capture the phone pose trajectory using ARKit). After face-alignment, the reconstructed face images using StyleGAN are as follows.

<video width="100%" controls>
  <source src="assets/video/independent-low.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/independent-low.mp4">here</a>.
</video>

The reconstructions are of varying quality, the identity seems to vary a bit, and the freeze frames are due to failed reconstructions for some of the frames (nearest neighbour shown in that case; failures mostly due to failing face alignment for tilted faces).

The next video shows the GP interpolation result, where we *only use the first and last frames* in the video and synthesise the rest using a view-aware GP prior in the latent space (using the pose trajectory from ARKit).

<video width="100%" controls>
  <source src="assets/video/gp-interpolation-low.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/gp-interpolation-low.mp4">here</a>.
</video>

From the GP model we can also directly get the marginal variance of the latent space predictions. Using sampling, we project that uncertainty to the image space, which is visualized in the video below. The uncertainty is low for the first and last frame (those are the observations!), but higher far from the known inputs.

<video width="100%" controls>
  <source src="assets/video/gp-uncertainty-low.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/gp-uncertainty-low.mp4">here</a>.
</video>

Finally, below are a set of frames summarising the differences between the independent, a naive linear, and the GP interpolated views.

![](assets/fig/face-synthesis.jpg)

Row #1: Frames separated by equal time intervals from a camera run, aligned on the face. Row #2: Each frame independently projected to GAN latent space and reconstructed. Row #3: Frames produced by reconstructing the first and the last frame and linearly interpolating the intermediate frames in GAN latent space. Row #4: Frames produced by reconstructing the first and the last frame, but interpolating the intermediate frames in GAN latent space by our view-aware GP prior. It can be seen that although linear interpolation achieves good quality, the azimuth rotation angle of the face is lost, as expected. With the view-aware prior, the rotation angle is better preserved. Row #5: The per-pixel uncertainty visualized in the form of standard deviation of the prediction at the corresponding time step. Heavier shading indicates higher uncertainty around the mean trajectory.




