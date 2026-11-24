"""
Copy a RootPainter project to a new project for an experiment.

The new project gets the same dataset, image order, classes (names and
brush colours) and annotations as the original, but no models, logs or
segmentations, so training starts from the beginning. Optionally a
models/model_settings.json is written with the input normalization and
augmentation to use.

Brush colours must match exactly, because the trainer tells classes apart
by colour. That is why the project is copied instead of created again in
the painter.

The original project is only read, never changed.

Example (Windows, Anaconda prompt):

    python C:\\root_painter_mappainter\\geo\\copy_project.py ^
        --syncdir C:\\root_painter_sync ^
        --source my_project --target my_project_imagenet_ambio ^
        --normalization imagenet --augmentation ambio
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath

NORMALIZATION_MODES = ('tile', 'imagenet')
AUGMENTATION_MODES = ('rootpainter', 'ambio')


def find_project_file(project_dir):
    proj_files = sorted(project_dir.glob('*.seg_proj'))
    if len(proj_files) != 1:
        sys.exit(f'Expected one .seg_proj file in {project_dir}, '
                 f'found {len(proj_files)}')
    return proj_files[0]


def copy_project(sync_dir, source, target, normalization=None,
                 augmentation=None):
    sync_dir = Path(sync_dir)
    source_dir = sync_dir / 'projects' / source
    target_dir = sync_dir / 'projects' / target
    if not source_dir.is_dir():
        sys.exit(f'Source project not found: {source_dir}')
    if target_dir.exists():
        sys.exit(f'Target project already exists, not overwriting: {target_dir}')

    with open(find_project_file(source_dir), 'r') as json_file:
        source_info = json.load(json_file)

    for sub in [('annotations', 'train'), ('annotations', 'val'),
                ('segmentations',), ('models',), ('messages',), ('logs',)]:
        os.makedirs(target_dir.joinpath(*sub))

    counts = {}
    for split in ('train', 'val'):
        src = source_dir / 'annotations' / split
        dst = target_dir / 'annotations' / split
        counts[split] = 0
        if src.is_dir():
            for fname in os.listdir(src):
                if fname.startswith('.'):
                    continue
                if (src / fname).is_file():
                    shutil.copy2(src / fname, dst / fname)
                    counts[split] += 1

    target_info = dict(source_info)
    target_info['name'] = target
    target_info['location'] = str(PurePosixPath('projects', target))
    target_info['original_model_file'] = 'random weights'
    target_info['copied_from_project'] = source
    with open(target_dir / (target + '.seg_proj'), 'w') as json_file:
        json.dump(target_info, json_file, indent=4)

    settings = {}
    if normalization:
        settings['normalization'] = normalization
    if augmentation:
        settings['augmentation'] = augmentation
    if settings:
        with open(target_dir / 'models' / 'model_settings.json', 'w') as f:
            json.dump(settings, f, indent=4)

    classes = source_info.get('classes')
    print(f'Created {target_dir}')
    print(f'  annotations copied: {counts["train"]} train, {counts["val"]} val')
    if classes:
        print('  classes: ' + ', '.join(c[0] for c in classes))
    else:
        print('  classes: none stored (old red/green project)')
    print(f'  settings: {settings if settings else "none (original behaviour)"}')
    print('Open the new project in the painter and start training.')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--syncdir', required=True,
                        help='RootPainter sync folder, e.g. C:\\root_painter_sync')
    parser.add_argument('--source', required=True,
                        help='name of the existing project (folder in projects)')
    parser.add_argument('--target', required=True,
                        help='name for the new project')
    parser.add_argument('--normalization', choices=NORMALIZATION_MODES,
                        help='input normalization for the new project')
    parser.add_argument('--augmentation', choices=AUGMENTATION_MODES,
                        help='data augmentation for the new project')
    args = parser.parse_args()
    copy_project(args.syncdir, args.source, args.target,
                 args.normalization, args.augmentation)


if __name__ == '__main__':
    main()
