# InsetGAN &mdash; Official PyTorch implementation

![Teaser image](./docs/insetgan_applications.jpg)


<a href='https://arxiv.org/abs/2203.07293'>**InsetGAN for Full-Body Image Generation**</a><br>
****Anna Frühstück, Krishna Kumar Singh, Eli Shechtman, Niloy Mitra, Peter Wonka, Jingwan Lu****<br>
***published at CVPR 2022***<br>
[Project Webpage](afruehstueck.github.io/insetgan)

**Abstract**
While GANs can produce photo-realistic images in ideal conditions for certain domains, the generation of full-body human images remains difficult due to the diversity of identities, hairstyles, clothing, and the variance in pose. Instead of modeling this complex domain with a single GAN, we propose a novel method to combine multiple pretrained GANs, where one GAN generates a global canvas (e.g., human body) and a set of specialized GANs, or insets, focus on different parts (e.g., faces, shoes) that can be seamlessly inserted onto the global canvas. We model the problem as jointly exploring the respective latent spaces such that the generated images can be combined, by inserting the parts from the specialized generators onto the global canvas, without introducing seams. We demonstrate the setup by combining a full body GAN with a dedicated high-quality face GAN to produce plausible-looking humans. We evaluate our results with quantitative metrics and user studies.

## Code
Our code is using models trained using the <a href='https://github.com/NVlabs/stylegan2-ada-pytorch'>`stylegan2-ada-pytorch` implementation by Nvidia</a>.


## Pre-trained Models
**Full-body humans**
We provide a pretrained model for generating full-body humans at 1024×768px resolution trained on a subset of images from the <a href='http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html'>DeepFashion dataset</a>.

[» Download link «](https://www.dropbox.com/s/e9wf6e8mle4ifzf/DeepFashion_1024x768.pkl)


**Insets**
You can use the pretrained FFHQ face generator< as an inset for the face region.

[» Download link «](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl)


## Generate your own Human Dataset
Given a database of human images, you can estimate the position of the human body within their image. Given a segmentation mask, you can blur the background.

## Train your own Models

In order to train your own models for canvas and/or insets, use the code provided by Nvidia in their `stylegan2-ada-pytorch` repository.  
Essentially, you provide your training data in a `.zip` file and train your network using
```
python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1
```
**Training Configuration**
Our models were trained with ADA augmentation, x-flips, and an empirically chosen R1-gamma value (R1=13.1 for our own 1024×1024px human generator, R1=0.2 for our 256×256px face generator).

## Citation
```
@inproceedings{Fruehstueck2022InsetGAN,
  title = {InsetGAN for Full-Body Image Generation},
  author = {Fr{\"u}hst{\"u}ck, Anna and Singh, {Krishna Kumar} and Shechtman, Eli and Mitra, {Niloy J.} and Wonka, Peter and Lu, Jingwan},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
```

## License
This work is made available under the [Something Something License]().

## Acknowledgements
This project started as an internship by Anna Frühstück with Adobe Research.

We thank the StyleGAN2-ADA team at Nvidia for providing their code.