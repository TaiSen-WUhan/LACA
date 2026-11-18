# Datasets
Download the datasets required by the selected experiment. Set the shared parent with `--data-root /path/to/datasets` or `DATASETS.ROOT`; the default is the project's `datasets/` directory.

<table style="border-collapse: collapse; width: 80%;">
<tr>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo" target="_blank">KITTI-2012</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo" target="_blank">KITTI-2015</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://vision.middlebury.edu/stereo/submit3/" target="_blank">Middlebury</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://www.eth3d.net/datasets" target="_blank">ETH3D</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://github.com/YuhuaXu/StereoDataset" target="_blank">InStereo2K</a></td>
  </tr>
    <tr>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/" target="_blank">Virtual KITTI 2</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html" target="_blank">SceneFlow</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://github.com/castacks/tartanair_tools" target="_blank">TartanAir</a>
</td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://github.com/megvii-research/CREStereo" target="_blank">CREStereo Dataset</a>
</td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation" target="_blank">FallingThings</a>
</td>
  </tr>
  <tr>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="http://sintel.is.tue.mpg.de/stereo" target="_blank">Sintel Stereo</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view" target="_blank">HR-VS</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://github.com/sniklaus/3d-ken-burns" target="_blank"><del>3D Ken Burns</del></a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://github.com/HKBU-HPML/IRS" target="_blank"><del>IRS Dataset</del></a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://cvlab-unibo.github.io/booster-web/" target="_blank">Booster Dataset</a></td>
  </tr>
</table>
The dataset are organized as follows,

```
LACA
└── datasets
    ├── booster
    │   ├── test
    │   └── train
    ├── CREStereo
    │   ├── hole
    │   ├── reflective
    │   ├── shapenet
    │   └── tree
    ├── ETH3D
    │   ├── two_view_testing
    │   ├── two_view_training
    │   └── two_view_training_gt
    ├── FallingThings
    │   ├── single
    │   └── mixed
    ├── carla-highres
    │   └── trainingF
    ├── InStereo2K
    │   ├── train
    │   └── test
    ├── KITTI
    |   ├── KITTI_2012
    │   |   ├── testing
    │   |   └── training
    |   └── KITTI_2015
    │       ├── testing
    │       └── training
    ├── Middlebury
    │   ├── 2005
    │   ├── 2006
    │   ├── 2014
    │   ├── 2021
    │   └── MiddEval3
    ├── SceneFlow
    │   ├── Driving
    │   ├── FlyingThings3D
    │   └── Monkaa
    ├── SintelStereo 
    │   └── training
    ├── TartanAir
    │   ├── abandonedfactory
    │   ├── abandonedfactory_night
    │   └── ...
    └── vkitti 
        ├── Scene01
        ├── Scene02
        ├── Scene06
        ├── Scene18
        └── Scene20
```
For Scene Flow, each of `FlyingThings3D`, `Monkaa`, and `Driving` contains `frames_finalpass/` and `disparity/`. FlyingThings3D uses `TRAIN/` and `TEST/` with the original scene folders and `left/`, `right/` subfolders. The alternative folder spelling `Flyingthings3d` is also accepted.

KITTI 2012 uses `colored_0/`, `colored_1/`, and `disp_occ/` beneath `training/`; KITTI 2015 uses `image_2/`, `image_3/`, and `disp_occ_0/`. The supplied split lists in the project `filenames/` directory are resolved automatically. Test submission commands read the corresponding `testing/` folders.

Middlebury evaluation uses `MiddEval3/trainingF/`, `trainingH/`, or `trainingQ/`, each containing scene folders with `im0.png`, `im1.png`, `disp0GT.pfm`, and `mask0nocc.png`. Submission commands read `testF/`, `testH/`, or `testQ/`. ETH3D uses `two_view_training/` with ground truth in `two_view_training_gt/`, and `two_view_testing/` for submissions.
