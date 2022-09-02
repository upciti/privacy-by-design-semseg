# On the Design of Privacy-Aware Cameras: A Study on Deep Neural Networks

This repository contains Pytorch training and evaluation code for semantic segmentation using DeepLabv3. It was partially inspired from [aerial_mtl][mtl-github].

For details, see [On the Design of Privacy-Aware Cameras: a Study on Deep Neural Networks][arxiv-paper].

This paper was accepted on [ECCV 2022][eccv2022] [IWDSC][iwdsc] (International Workshop on Distributed Cameras).

If you use this code for a paper, please cite: 

```shell
@article{carvalho2022privacy,
  title={On the Design of Privacy-Aware Cameras: a Study on Deep Neural Networks},
  author={Marcela Carvalho and Oussama Ennaffi and Sylvain Chateau and Samy Ait Bachir},
  journal={available-soon},
  year={2022}
}
```
Code will be available soon.

## Usage
Clone this repository locally:
```shell
git clone https://github.com/upciti/privacy-by-design-semseg
```

We use poetry to... so, to install all the dependencies, we suggest using python ... and pyenv here is the explanation.

Compared to conda, which is the togo solution in the field, poetry has the advantage of... However there are some problems when we use some library that is not on their repo.

```shell
poetry install  # reads the pyptoject.toml file, resolves dependencies and installs them
poetry shell  # to activate the virtual environment in the current shell
poe poe-torch-cuda11  # to install Pytorch with CUDA 11.6
```

## Data preparation

To be added.

[//]: # (To generate out-of-focus dataset, you must download ... image from Cityscape and disparity maps.)

[//]: # ()
[//]: # (To convert disparity maps to depth maps, we use:)

[//]: # ()
[//]: # (```shell)

[//]: # (this to convert to depth maps)

[//]: # (```)

[//]: # ()
[//]: # (Then, these depth maps are used with this matlab/octave code to generate defocused images using the method... The original code was written by ... and ... in ...)

[//]: # ()
[//]: # (run xyz to generate the dataset. Change the parameters according to the experiment in the paper, or personal usage.)

[//]: # ()
[//]: # (Then, define the following environment variable to the )

## Model zoo

| name               | url                                         |
|--------------------|---------------------------------------------|
| infocus_color      | [download_weight][infocus_color-model-link] |
| infocus_gray       | [download_weight][infocus_gray-model-link]  |
| defocus_color      | [download_weight][defocus-color-link]       |
| defocus_gray       | [download_weight][defocus_gray-model-link]  |

We suggest to save weights inside a folder called ```checkpoints/model_name```.

For the command lines we mention in this README, we consider you saved ```infocus_color``` weights in ```./checkpoints/infocus_color/0300.pth.tar```.


## Visualisation
We adopt visdom to visualise training artifacts and metrics results. To run visdom, in another shell inside the project folder, run:

```shell
poetry shell
visdom -p $DISPLAY_PORT
```


## Training
To train a pretrained DeepLabV3 with one of our weights with a single GPU, run: 

```shell
python train_test_semseg.py --train --name name-of-the-new-project-with-pretrained-weights --dataroot PATH_TO_CITYSCAPES --batchSize $batch_size --nEpochs $end --display_id $display_id --port $port --use_resize --data_augmentation f f --resume
```

To train from scratch, run:

```shell
python train_test_semseg.py --train --name name-of-the-new-project --dataroot PATH_TO_CITYSCAPES --batchSize $batch_size --nEpochs $end --display_id $display_id --port $port --use_resize --data_augmentation f f 
```


## Evaluation
To train a pretrained DeepLabV3 with one of our weights with a single GPU, run: 

Generate images only:
```shell
python train_test_semseg.py --test --test_only --name infocus_color --save_samples --use_resize --display_id 0 --dataroot PATH_TO_DATA
```


Generate images and evaluation:
```shell
python train_test_semseg.py --test --test_metrics --name infocus_color --use_resize --display_id 0 --dataroot ./datasets/public_datasets/Cityscapes
```

As a result from this last run, you should get the outputs under the results file and the following metric results:
```
OA:  0.9387813619099232
AA:  0.7375342230955388
mIOU 0.6486886632431875
```

Generate only evaluation (resulting segmentation must be in the corresponding folder):
```shell
python train_test_semseg.py --test --evaluate_only --name infocus_color --use_resize --display_id 0 --dataroot ./datasets/public_datasets/Cityscapes
```

## License
Code (scripts) are released under the [MIT license][license].

[//]: # (References)

[mtl-github]: https://github.com/marcelampc/aerial_mtl
[iwdsc]: https://iwdsc.github.io/
[eccv2022]: https://eccv2022.ecva.net/
[arxiv-paper]: https://arxiv.org/list/cs.CV/recent
[model-link]: broken
[license]: LICENSE
[infocus_color-model-link]: https://upciti-computer-vision-public.s3.eu-west-3.amazonaws.com/weights-privacy-aware-paper/infocus_color/0300.pth.tar 
[infocus_gray-model-link]: https://upciti-computer-vision-public.s3.eu-west-3.amazonaws.com/weights-privacy-aware-paper/infocus_gray/0300.pth.tar
[defocus-color-link]: https://upciti-computer-vision-public.s3.eu-west-3.amazonaws.com/weights-privacy-aware-paper/defocus_color/0300.pth.tar
[defocus_gray-model-link]: https://upciti-computer-vision-public.s3.eu-west-3.amazonaws.com/weights-privacy-aware-paper/defocus_gray/0300.pth.tar
