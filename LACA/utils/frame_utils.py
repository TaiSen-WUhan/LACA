from os.path import splitext
import re
import numpy as np
from PIL import Image
from os.path import *
import torch.nn.functional as F
import json
import imageio
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    with open(file, 'rb') as stream:
        header = stream.readline().rstrip()
        if header not in (b'PF', b'Pf'):
            raise ValueError(f'Not a PFM file: {file}')
        color = header == b'PF'
        dimensions = stream.readline().split()
        if len(dimensions) != 2:
            raise ValueError(f'Malformed PFM dimensions: {file}')
        width, height = map(int, dimensions)
        scale = float(stream.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(stream, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(data.reshape(shape))


def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
        os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())
        

def readDispKITTI(filename):
    # imdecode/fromfile also supports non-ASCII paths on Windows.
    image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise ValueError(f"Cannot read KITTI disparity: {filename}")
    disp = image.astype(np.float32) / 256.0
    valid = disp > 0.0
    return disp, valid


def readDispMiddlebury(file_name):
    if basename(file_name) == 'disp0GT.pfm':
        disp = readPFM(file_name).astype(np.float32)
        assert len(disp.shape) == 2
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        return disp, nocc_pix
    elif basename(file_name) == 'disp0.pfm':
        disp = readPFM(file_name).astype(np.float32)
        valid = disp < 1e3
        return disp, valid
    elif splitext(file_name)[-1] == '.png':
        disp = np.array(Image.open(file_name)).astype(np.float32)
        valid = disp > 0.0
        return disp, valid
    

def writeDispKITTI(filename, disp):
    disp = np.clip(np.round(disp * 256), 0, 65535).astype(np.uint16)
    success, encoded = cv2.imencode('.png', disp)
    if not success:
        raise OSError(f"Cannot encode KITTI disparity: {filename}")
    encoded.tofile(filename)
    
    
def readDispVKITTI(filename):
    depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    f = 725.0087
    baseline = 0.532725
    disp = f * baseline * 100 / depth
    # magic value = f * baseline * 100 / 65535 = 0.589
    valid = disp > 0.59
    return disp, valid

    
# Method taken from /n/fs/raft-depth/RAFT-Stereo/datasets/SintelStereo/sdk/python/sintel_io.py
def readDispSintelStereo(file_name):
    a = np.array(Image.open(file_name))
    d_r, d_g, d_b = np.split(a, axis=2, indices_or_sections=3)
    disp = (d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14))[..., 0]
    # mask = np.array(Image.open(file_name.replace('disparities', 'occlusions')))
    # valid = ((mask == 0) & (disp > 0))
    valid = disp > 0
    return disp, valid


# Method taken from https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt
def readDispFallingThings(file_name):
    a = np.array(Image.open(file_name))
    with open(join(dirname(file_name), '_camera_settings.json'), 'r') as f:
        intrinsics = json.load(f)
    fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    disp = (fx * 6.0 * 100) / a.astype(np.float32)
    valid = disp > 0
    return disp, valid

    
# Method taken from https://github.com/castacks/tartanair_tools/blob/master/data_type.md
def readDispTartanAir(file_name):
    depth = np.load(file_name)
    disp = 80.0 / depth
    valid = disp > 0
    return disp, valid


def readDispBooster(file_name):
    disp = np.load(file_name)
    valid = disp > 0
    return disp, valid


def readDispCREStereo(file_name):
    disp = np.asarray(Image.open(file_name), dtype=np.float32)
    disp = disp / 32.0
    valid = disp > 0
    return disp, valid


def readDispInStereo2K(file_name):
    disp = np.asarray(Image.open(file_name), dtype=np.float32)
    disp = disp / 100.0
    valid = disp > 0
    return disp, valid


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1].lower()
    if ext in ('.png', '.jpeg', '.ppm', '.jpg', '.bmp', '.tif', '.tiff'):
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'nmrf':
            self._pad = [0, pad_wd, pad_ht, 0]
        else:
            raise ValueError(f"Non recognized mode '{mode}'")

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
