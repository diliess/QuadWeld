"""Rebuild a saved HJ-90 scan from RGB frames/calibration only, without Isaac or truth."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hj90.optics import reconstruct, rasterize, write_ply


def rebuild(directory, output=None):
    directory = Path(directory)
    output = Path(output or directory/'reconstructed_from_rgb.npz')
    camera = json.loads((directory/'camera.json').read_text())
    c = camera['optics']
    frames = json.loads((directory/'frames.json').read_text())
    static = (directory/'left_off.png').is_file()
    off = [np.asarray(Image.open(directory/(s+'_off.png'))) for s in ('left','right')] if static else None
    points, world, uv, confidence, ids = [], [], [], [], []
    for frame in frames:
        with np.load(directory/'frames'/f"{frame['index']:04d}.npz", allow_pickle=False) as raw:
            pair = off if static else [raw['left_off'],raw['right_off']]
            xyz,xy,strength = reconstruct(raw['left_on'],pair[0],raw['right_on'],pair[1],c,
                                          np.radians(frame['laser_angle_deg']))
        transform = np.array(frame['world_from_mid_optical'])
        points.append(xyz)
        world.append(xyz @ transform[:3,:3].T + transform[:3,3])
        uv.append(xy)
        confidence.append(strength)
        ids.append(np.full(len(xyz),frame['index'],np.int32))
    xyz = np.concatenate(points)
    payload = {'points_mid_optical_m':xyz.astype(np.float32),'points_world_m':np.concatenate(world).astype(np.float32),
               'left_uv':np.concatenate(uv).astype(np.float32),'confidence':np.concatenate(confidence).astype(np.float32),
               'frame_index':np.concatenate(ids)}
    if static:
        depth = rasterize(xyz,c)
        payload.update(depth_m=depth,valid_depth_mask=np.isfinite(depth))
    np.savez_compressed(output,**payload)
    write_ply(output.with_suffix('.ply'),payload['points_world_m'])
    print('RGB_ONLY_RECONSTRUCTION',len(xyz),'points:',output)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    rebuild(a.directory,a.output)
