"""Update the portable complete scene with a functional HJ-90 stereo/laser rig."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hj90.optics import load_calibration
from hj90.usd import export_layer


def build():
    bundle = ROOT/'assets/scenes/industrial-welding'
    info = export_layer(bundle, load_calibration())
    manifest = json.loads((bundle/'manifest.json').read_text())
    manifest['full_scene']['active_stereo_laser'] = {k:v for k,v in info.items() if k!='extra_files'}
    names = sorted({f['path'] for f in manifest['files']} | set(info['extra_files']))
    manifest['files'] = [{'path':n, 'bytes':(bundle/n).stat().st_size,
                          'sha256':hashlib.sha256((bundle/n).read_bytes()).hexdigest()} for n in names]
    manifest['sensor_scope'] = 'two independent pinhole cameras and RTX line projector; RGB-only scan reconstruction; unmeasured calibration assumptions'
    (bundle/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print('HJ90_SENSOR_LAYER', bundle/info['layer'])


if __name__=='__main__':
    build()
