import logging
from copy import deepcopy
import os
from pathlib import Path
from glob import glob
import os.path as osp
import numpy as np
import torch
import torch.utils
import torch.nn.functional as F
from typing import Iterable, Optional, TypeVar, List, Tuple

from ..utils import frame_utils, misc, dist_utils as comm
from .transforms import FlowAugmentor, SparseFlowAugmentor
from .base.easy_dataset import EasyDataset
from .sampler import InferenceSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def dataset_root(cfg, name):
    """Resolve every dataset beneath the shared DATASETS.ROOT directory."""
    return str(Path(cfg.DATASETS.ROOT) / name)


def read_rgb(path):
    return np.array(frame_utils.read_gen(str(path)).convert("RGB"), dtype=np.uint8)


def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


T = TypeVar("T", str, bytes)


def verify_str_arg(
    value: T,
    arg: Optional[str] = None,
    valid_values: Optional[Iterable[T]] = None,
    custom_msg: Optional[str] = None,
) -> T:
    if not isinstance(value, str):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)
    
    if valid_values is None:
        return value
    
    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = "Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}."
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)
    
    return value


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as fp:
        lines = [line.rstrip() for line in fp.readlines()]
    return lines


class StereoDataset(EasyDataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        aug_params = deepcopy(aug_params)
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if self.sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.disparity_list = []
        self.image_list = []

    def __getitem__(self, index):

        sample = {}
        if self.is_test:
            img1, img2 = (read_rgb(path) for path in self.image_list[index])
            if img1.shape != img2.shape:
                raise ValueError(f"Stereo images must have equal dimensions: {self.image_list[index]}")
            sample['img1'] = torch.from_numpy(img1).permute(2, 0, 1).float()
            sample['img2'] = torch.from_numpy(img2).permute(2, 0, 1).float()
            sample['meta'] = self.image_list[index][0]
            return sample
        
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            initial_seed = torch.initial_seed() % 2**31
            if worker_info is not None:
                misc.seed_all_rng(initial_seed + worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp_path = self.disparity_list[index]
        if isinstance(disp_path, (tuple, list)):
            disp_path = disp_path[0]
        disp = self.disparity_reader(disp_path)
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1, img2 = (read_rgb(path) for path in self.image_list[index])
        disp = np.array(disp, dtype=np.float32)
        if img1.shape != img2.shape or disp.shape != img1.shape[:2]:
            raise ValueError(f"Image/disparity sizes differ: {self.image_list[index]}")
        valid = np.asarray(valid, dtype=bool) & np.isfinite(disp) & (disp > 0)
        disp[~valid] = 0
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        sample['img1'] = torch.from_numpy(img1).permute(2, 0, 1).float()
        sample['img2'] = torch.from_numpy(img2).permute(2, 0, 1).float()
        sample['disp'] = torch.from_numpy(flow).permute(2, 0, 1).float()[0]

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (sample['disp'] > 0) & (sample['disp'] < 512)
        sample['valid'] = valid.bool()

        if self.img_pad is not None:
            padH, padW = self.img_pad
            padding = [padW] * 2 + [padH] * 2
            for key in ("img1", "img2", "disp", "valid"):
                sample[key] = F.pad(sample[key], padding)
        
        return sample
    
    def __len__(self):
        return len(self.image_list)
    
    def _scan_pairs(
        self,
        paths_left_pattern: str,
        paths_right_pattern: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        
        left_paths = list(sorted(glob(paths_left_pattern)))
        
        if paths_right_pattern:
            right_paths = list(sorted(glob(paths_right_pattern)))
        else:
            right_paths = list(None for _ in left_paths)
        
        if not left_paths:
            raise FileNotFoundError(f"Could not find any files matching the patterns: {paths_left_pattern}")
        
        if not right_paths:
            raise FileNotFoundError(f"Could not find any files matching the patterns: {paths_right_pattern}")
        
        if len(left_paths) != len(right_paths):
            raise ValueError(
                f"Found {len(left_paths)} left files but {len(right_paths)} right files using:\n"
                f"left pattern: {paths_left_pattern}\n"
                f"right pattern: {paths_right_pattern}\n"
            )
        
        paths = list((left, right) for left, right in zip(left_paths, right_paths))
        return paths
        
class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SceneFlow', dstype='frames_finalpass', things_test=False):
        super().__init__(aug_params)
        root = Path(root)
        # Both spellings occur in public Scene Flow downloads.
        things = root / 'FlyingThings3D'
        if not things.is_dir():
            things = root / 'Flyingthings3d'
        split = 'TEST' if things_test else 'TRAIN'
        self._add_pairs(things, dstype, f'{split}/*/*/left/*.png')
        if not things_test:
            self._add_pairs(root / 'Monkaa', dstype, '*/left/*.png')
            self._add_pairs(root / 'Driving', dstype, '*/*/*/left/*.png')

    def _add_pairs(self, root, dstype, pattern):
        start = len(self.image_list)
        for left in sorted((root / dstype).glob(pattern)):
            if left.stem.endswith('_lsc_lbl'):
                continue
            right = left.parent.parent / 'right' / left.name
            disparity = (root / 'disparity' / left.relative_to(root / dstype)).with_suffix('.pfm')
            self.image_list.append([str(left), str(right)])
            self.disparity_list.append(str(disparity))
        logging.getLogger(__name__).info('Added %d pairs from %s', len(self.image_list) - start, root)


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training'):
        super().__init__(aug_params, sparse=True)
        verify_str_arg(split, 'split', ['training', 'testing'])
        self.is_test = split == 'testing'
        for left in sorted((Path(root) / f'two_view_{split}').glob('*/im0.png')):
            self.image_list.append([str(left), str(left.with_name('im1.png'))])
            if not self.is_test:
                self.disparity_list.append(str(Path(root) / 'two_view_training_gt' / left.parent.name / 'disp0GT.pfm'))


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', split='training', image_set='kitti_mix'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        if split == 'testing':
            self.is_test = True
            if image_set == 'kitti_2012':
                root = osp.join(root, 'KITTI_2012')
                images1 = sorted(glob(osp.join(root, 'testing', 'colored_0/*_10.png')))
                images2 = sorted(glob(osp.join(root, 'testing', 'colored_1/*_10.png')))
            elif image_set == 'kitti_2015':
                root = osp.join(root, 'KITTI_2015')
                images1 = sorted(glob(osp.join(root, 'testing', 'image_2/*_10.png')))
                images2 = sorted(glob(osp.join(root, 'testing', 'image_3/*_10.png')))
            else:
                raise ValueError("Unknown dataset for test: '{}'".format(image_set))
            if len(images1) != len(images2):
                raise ValueError(f"KITTI left/right image counts differ in {root}")
            for img1, img2 in zip(images1, images2):
                if Path(img1).name != Path(img2).name:
                    raise ValueError(f"KITTI image pair does not match: {img1}, {img2}")
                self.image_list += [ [img1, img2] ]

        else:
            kitti_dict = {
                'kitti_mix_2012_train': 'filenames/KITTI_mix_2012_train.txt',
                'kitti_mix_2015_train': 'filenames/KITTI_mix_2015_train.txt',
                'kitti_2012_val': 'filenames/KITTI_2012_val.txt',
                'kitti_2015_val': 'filenames/KITTI_2015_val.txt',
                'kitti_mix': 'filenames/KITTI_mix.txt',
                'kitti_2015_train': 'filenames/KITTI_2015_train.txt',
                'kitti_2015_trainval': 'filenames/KITTI_2015_trainval.txt',
                'kitti_2012_train': 'filenames/KITTI_2012_train.txt',
                'kitti_2012_trainval': 'filenames/KITTI_2012_trainval.txt',
            }

            assert image_set in kitti_dict.keys()
            data_filename = PROJECT_ROOT / kitti_dict[image_set]

            self._root_12 = os.path.join(root, 'KITTI_2012')
            self._root_15 = os.path.join(root, 'KITTI_2015')

            self.load_path(data_filename)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        for line in splits:
            left_name = line[0].split('/')[1]
            if left_name.startswith('image'):
                root = self._root_15
            else:
                root = self._root_12
            img1 = os.path.join(root, line[0])
            img2 = os.path.join(root, line[1])
            self.image_list += [ [img1, img2] ]
            if len(line) > 2:
                disp = os.path.join(root, line[2])
                self.disparity_list += [ disp ]
            
            
class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Middlebury', split='F', image_set='training'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        assert split in ["F", "H", "Q", "2005", "2006", "2014", "2021"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [ [str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")] ]
                        self.disparity_list += [ str(scene / "disp1.png") ]
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [ [str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")] ]
                        self.disparity_list += [ str(scene / "disp1.png") ]  
        elif split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        elif split == "2021":
            scenes = list((Path(root) / "2021").glob("*"))
            for scene in scenes:
                self.image_list += [ [str(scene / "im0.png"), str(scene / "im1.png")] ]
                self.disparity_list += [ str(scene / "disp0.pfm") ]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [ [str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")] ]
                        self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            verify_str_arg(image_set, 'image_set', ['training', 'test'])
            self.is_test = image_set == 'test'
            for left in sorted((Path(root) / 'MiddEval3' / f'{image_set}{split}').glob('*/im0.png')):
                self.image_list.append([str(left), str(left.with_name('im1.png'))])
                if not self.is_test:
                    self.disparity_list.append(str(left.with_name('disp0GT.pfm')))


class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)
        
        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2
        
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert Path(img1).parts[-2:] == Path(disp).parts[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
    
    
class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings', variant: str = "both"):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        root = Path(root)
        
        verify_str_arg(variant, "variant", valid_values=["single", "mixed", "both"])
        
        variants = {
            "single": ["single"],
            "mixed": ["mixed"],
            "both": ["single", "mixed"],
        }[variant]
        
        split_prefix = {
            "single": Path("*") / "*",
            "mixed": Path("*"),
        }
        
        for s in variants:
            left_img_pattern = str(root / s / split_prefix[s] / "*.left.jpg")
            right_img_pattern = str(root / s / split_prefix[s] / "*.right.jpg")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / split_prefix[s] / "*.left.depth.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)
               

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/TartanAir'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        root = Path(root)
        
        left_img_pattern = str(root / "*/*/*/image_left/*_left.png")
        right_img_pattern = str(root / "*/*/*/image_right/*_right.png")
        self.image_list = self._scan_pairs(left_img_pattern, right_img_pattern)
        
        left_disparity_pattern = str(root / "*/*/*/depth_left/*_left_depth.npy")
        self.disparity_list = self._scan_pairs(left_disparity_pattern, None)
        
        
class CREStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/CREStereo'):
        super().__init__(aug_params, reader=frame_utils.readDispCREStereo)
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        root = Path(root)
        
        dirs = ["shapenet", "reflective", "tree", "hole"]
        
        for s in dirs:
            left_img_pattern = str(root / s / "*_left.jpg")
            right_img_pattern = str(root / s / "*_right.jpg")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / "*_left.disp.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)
            

class VirtualKitti2(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/vkitti'):
        super().__init__(aug_params, reader=frame_utils.readDispVKITTI)
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        root = Path(root)
        
        dirs = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        for s in dirs:
            left_img_pattern = str(root / s / "*" / "frames" / "rgb" / "Camera_0" / "rgb_*.jpg")
            right_img_pattern = str(root / s / "*" / "frames" / "rgb" / "Camera_1" / "rgb_*.jpg")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / "*" / "frames" / "depth" / "Camera_0" / "depth_*.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)


class CarlaHighres(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/carla-highres'):
        super().__init__(aug_params=aug_params)
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        image1_list = sorted(glob(osp.join(root, 'trainingF/*/im0.png')))
        image2_list = [im.replace('im0', 'im1') for im in image1_list]
        disp_list = [im.replace('im0.png', 'disp0GT.pfm') for im in image1_list]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/InStereo2K', variant: str = "both"):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        root = Path(root)
        
        verify_str_arg(variant, "variant", valid_values=["train", "test", "both"])
        
        variants = {
            "train": ["train"],
            "test": ["test"],
            "both": ["train", "test"],
        }[variant]
        
        for s in variants:
            left_img_pattern = str(root / s / "*" / "left.png")
            right_img_pattern = str(root / s / "*" / "right.png")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / "*" / "left_disp.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)
            

class Booster(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/booster', split='train'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispBooster)
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        folder_list = sorted(glob(osp.join(root, split + '/balanced/*')))        
        for folder in folder_list:
            image1_list = sorted(glob(osp.join(folder, 'camera_00/im*.png')))
            image2_list = sorted(glob(osp.join(folder, 'camera_02/im*.png')))
            if split == "train":
                for img1 in image1_list:
                    for img2 in image2_list:
                        self.image_list += [ [img1, img2] ]
                        self.disparity_list += [ osp.join(folder, 'disp_00.npy') ]
            else:
                for img1, img2 in zip(image1_list, image2_list):
                    self.image_list += [ [img1, img2] ]
                    self.disparity_list += [ osp.join(folder, 'disp_00.npy') ]


def build_train_loader(cfg):
    """ Create the data loader for the corresponding training set """
    crop_size = cfg.DATASETS.CROP_SIZE
    spatial_scale = cfg.DATASETS.SPATIAL_SCALE
    yjitter = cfg.DATASETS.YJITTER
    aug_params = {'crop_size': list(crop_size), 'min_scale': spatial_scale[0], 'max_scale': spatial_scale[1], 'do_flip': False, 'yjitter': yjitter}
    if cfg.DATASETS.SATURATION_RANGE is not None:
        aug_params["saturation_range"] = cfg.DATASETS.SATURATION_RANGE
    if cfg.DATASETS.IMG_GAMMA is not None and cfg.DATASETS.IMG_GAMMA > 0:
        aug_params["gamma"] = cfg.DATASETS.IMG_GAMMA
    if cfg.DATASETS.DO_FLIP is not None:
        aug_params["do_flip"] = cfg.DATASETS.DO_FLIP

    train_dataset = None
    logger = logging.getLogger(__name__)
    assert len(cfg.DATASETS.TRAIN) == len(cfg.DATASETS.MUL)
    for dataset_name, mul in zip(cfg.DATASETS.TRAIN, cfg.DATASETS.MUL):
        if dataset_name.startswith('middlebury_'):
            new_dataset = Middlebury(aug_params, root=dataset_root(cfg, 'Middlebury'), split=dataset_name.replace('middlebury_', ''))
            logger.info(f"{len(new_dataset)} samples from {dataset_name}")
        elif dataset_name == 'sceneflow':
            new_dataset = SceneFlowDatasets(aug_params, root=dataset_root(cfg, 'SceneFlow'), dstype='frames_finalpass')
            logger.info(f"{len(new_dataset)} samples from SceneFlow")
        elif dataset_name == 'eth3d':
            new_dataset = ETH3D(aug_params, root=dataset_root(cfg, 'ETH3D'), split='training')
            logger.info(f"{len(new_dataset)} samples from ETH3D")
        elif 'kitti_' in dataset_name:
            new_dataset = KITTI(aug_params, root=dataset_root(cfg, 'KITTI'), image_set=dataset_name)
            logger.info(f"{len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params, root=dataset_root(cfg, 'SintelStereo'))
            logger.info(f"{len(new_dataset)} samples from SintelStereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params, root=dataset_root(cfg, 'FallingThings'), variant='both')
            logger.info(f"{len(new_dataset)} samples from FallingThings")
        elif dataset_name == 'tartan_air':
            new_dataset = TartanAir(aug_params, root=dataset_root(cfg, 'TartanAir'))
            logger.info(f"{len(new_dataset)} samples from TartanAir")
        elif dataset_name == 'carla_highres':
            new_dataset = CarlaHighres(aug_params, root=dataset_root(cfg, 'carla-highres'))
            logger.info(f"{len(new_dataset)} samples from Carla Highres")
        elif dataset_name == 'crestereo':
            new_dataset = CREStereo(aug_params, root=dataset_root(cfg, 'CREStereo'))
            logger.info(f"{len(new_dataset)} samples from CREStereo")
        elif dataset_name == 'vkitti2':
            new_dataset = VirtualKitti2(aug_params, root=dataset_root(cfg, 'vkitti'))
            logger.info(f"{len(new_dataset)} samples from VirtualKitti2")
        elif dataset_name.startswith('booster'):
            new_dataset = Booster(aug_params, root=dataset_root(cfg, 'booster'))
            logger.info(f"{len(new_dataset)} samples from Booster")
        elif dataset_name == 'in_stereo2k':
            new_dataset = InStereo2K(aug_params, root=dataset_root(cfg, 'InStereo2K'), variant='both')
            logger.info(f"{len(new_dataset)} samples from InStereo2K")
        else:
            raise ValueError(f"Unrecognized dataset {dataset_name}")
        if len(new_dataset) == 0:
            raise ValueError(f"No samples found for {dataset_name} under {cfg.DATASETS.ROOT}")
        if mul > 0:
            new_dataset = mul * new_dataset
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    if train_dataset is None:
        raise ValueError("DATASETS.TRAIN must contain at least one dataset.")
    world_size = comm.get_world_size()
    total_batch_size = cfg.SOLVER.IMS_PER_BATCH
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({})".format(
        total_batch_size, world_size
    )
    batch_size = cfg.SOLVER.IMS_PER_BATCH // world_size

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=comm.get_world_size(),
            rank=comm.get_rank())
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)
    
    return train_loader, train_sampler


def build_val_loader(cfg, dataset_name):
    logger = logging.getLogger(__name__)
    if dataset_name == 'things':
        val_dataset = SceneFlowDatasets(root=dataset_root(cfg, 'SceneFlow'), dstype='frames_finalpass', things_test=True)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif 'kitti' in dataset_name:
        # perform validation using the KITTI (train) split
        val_dataset = KITTI(root=dataset_root(cfg, 'KITTI'), image_set=dataset_name)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name == 'eth3d':
        val_dataset = ETH3D(root=dataset_root(cfg, 'ETH3D'), split='training')
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name.startswith("middlebury_"):
        val_dataset = Middlebury(root=dataset_root(cfg, 'Middlebury'), split=dataset_name.replace('middlebury_', ''))
        logger.info('Number of validation image pairs: %d' % len(val_dataset))

    else:
        raise ValueError(f"Unsupported evaluation dataset: {dataset_name}")
    if len(val_dataset) == 0:
        raise ValueError(f"No samples found for {dataset_name} under {cfg.DATASETS.ROOT}")

    world_size = comm.get_world_size()
    if world_size > 1:
        val_sampler = InferenceSampler(len(val_dataset))
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=True, drop_last=False,
                                             sampler=val_sampler)
    return val_loader