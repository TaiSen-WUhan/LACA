"""Run LACA on rectified stereo pairs or export benchmark submissions."""
import argparse
from glob import glob
import logging
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from LACA.checkpoint import load_checkpoint
from LACA.config.runtime import build_config
from LACA.dataloader import datasets
from LACA.laca import LACA
from LACA.utils import frame_utils
from LACA.utils.logger import setup_logger


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-pretrained', default='sf', help='Checkpoint path or sf/l_sf/kitti')
    parser.add_argument('--config-file', default='', help='Optional experiment YAML')
    parser.add_argument('--data-root', help='Parent directory containing KITTI, ETH3D, Middlebury, etc.')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset-name', choices=['kitti_2012', 'kitti_2015', 'eth3d',
                                                  'middlebury_F', 'middlebury_H', 'middlebury_Q'])
    source.add_argument('--input', nargs=2, metavar=('LEFT', 'RIGHT'),
                        help='Two files, directories, or quoted glob patterns, paired in sorted order')
    parser.add_argument('--output', help='Output directory (default: results for images, . for benchmarks)')
    parser.add_argument('opts', nargs=argparse.REMAINDER, help='Configuration KEY VALUE pairs')
    return parser


def collect_images(source):
    path = Path(source).expanduser()
    candidates = path.iterdir() if path.is_dir() else (Path(p) for p in glob(str(path), recursive=True))
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.ppm'}
    images = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in extensions)
    if not images:
        raise FileNotFoundError(f'No images found: {source}')
    return images


def stereo_pairs(left, right):
    left, right = collect_images(left), collect_images(right)
    if len(left) != len(right):
        raise ValueError(f'Left/right image counts differ: {len(left)} vs {len(right)}')
    dataset = datasets.StereoDataset()
    dataset.image_list = [(str(l), str(r)) for l, r in zip(left, right)]
    dataset.is_test = True
    return dataset


@torch.no_grad()
def predict(model, sample):
    device = next(model.parameters()).device
    inputs = {key: sample[key].unsqueeze(0).to(device) for key in ('img1', 'img2')}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    prediction = model(inputs)['disp_pred'][0]
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    disparity = prediction.cpu().numpy().astype(np.float32)
    if not np.isfinite(disparity).all():
        raise FloatingPointError(f'Non-finite prediction for {sample["meta"]}')
    return disparity, elapsed


def save_visualization(path, disparity):
    from matplotlib import colormaps

    low, high = float(disparity.min()), float(disparity.max())
    normalized = (disparity - low) / max(high - low, 1e-6)
    rgb = colormaps['turbo'](normalized, bytes=True)[..., :3]
    Image.fromarray(rgb).save(path)


def run_inference(model, dataset, output, dataset_name=None):
    if not len(dataset):
        raise ValueError('No stereo pairs found; check --data-root and the dataset split.')
    output = Path(output)
    if dataset_name in ('kitti_2012', 'kitti_2015'):
        output = output / f'{dataset_name}_submission'
    elif dataset_name == 'eth3d':
        output = output / 'low_res_two_view'
    if dataset_name is None:
        left_root = Path(os.path.commonpath([str(Path(pair[0]).resolve().parent) for pair in dataset.image_list]))
        names = [Path(pair[0]).resolve().relative_to(left_root).with_suffix('') for pair in dataset.image_list]
        if len(set(names)) != len(names):
            raise ValueError('Input images have duplicate output names; use a more specific input pattern.')
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('LACA')
    # Exclude the first CUDA/kernel initialization from reported runtime.
    predict(model, dataset[0])
    elapsed = 0.0
    for index in range(len(dataset)):
        sample = dataset[index]
        disparity, duration = predict(model, sample)
        elapsed += duration
        left = Path(sample['meta'])
        if dataset_name in ('kitti_2012', 'kitti_2015'):
            frame_utils.writeDispKITTI(str(output / left.name), disparity)
        elif dataset_name == 'eth3d':
            frame_utils.writePFM(str(output / f'{left.parent.name}.pfm'), disparity)
            (output / f'{left.parent.name}.txt').write_text(f'runtime {duration:.6f}\n', encoding='utf-8')
        elif dataset_name and dataset_name.startswith('middlebury_'):
            scene = output / left.parent.parent.name / left.parent.name
            scene.mkdir(parents=True, exist_ok=True)
            frame_utils.writePFM(str(scene / 'disp0LACA.pfm'), disparity)
            (scene / 'timeLACA.txt').write_text(f'{duration:.6f}\n', encoding='utf-8')
        else:
            stem = output / names[index]
            stem.parent.mkdir(parents=True, exist_ok=True)
            # NPY preserves floating-point disparity; PNG is for visualization only.
            np.save(str(stem) + '.npy', disparity)
            save_visualization(str(stem) + '.png', disparity)
        logger.info('[%d/%d] %s (%.3f s)', index + 1, len(dataset), left.name, duration)
    logger.info('Saved predictions to %s; average runtime %.4f s/pair.', output.resolve(), elapsed / len(dataset))


def main(args):
    setup_logger(name='LACA')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: cpu, cuda, cuda:N.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu.')
    checkpoint = load_checkpoint(args.from_pretrained)
    cfg = build_config(args.config_file, args.opts, checkpoint.get('model_config'), args.data_root)
    model = LACA(cfg, mono_pretrained=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model = model.to(device).eval()
    if args.input:
        dataset = stereo_pairs(*args.input)
    elif args.dataset_name.startswith('kitti_'):
        dataset = datasets.KITTI(root=datasets.dataset_root(cfg, 'KITTI'), split='testing', image_set=args.dataset_name)
    elif args.dataset_name == 'eth3d':
        dataset = datasets.ETH3D(root=datasets.dataset_root(cfg, 'ETH3D'), split='testing')
    else:
        dataset = datasets.Middlebury(root=datasets.dataset_root(cfg, 'Middlebury'),
                                      split=args.dataset_name.split('_')[1], image_set='test')
    run_inference(model, dataset, args.output or ('results' if args.input else '.'), args.dataset_name)


if __name__ == '__main__':
    main(get_parser().parse_args())
