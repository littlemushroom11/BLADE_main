"""Load the repository's clean split or full ResNet-18 checkpoints.

Repartitioning uses the trained server stages, never the unused randomly
initialized stages that older ResNetClient checkpoints also contain.
"""

from pathlib import Path

import torch


RESNET18_NAMES = {'resnet18', 'resnet18_cifar', 'cifar_resnet18'}
SERVER_STAGES = {
    'early': (2, 3, 4),
    'middle': (3, 4),
    'late': (4,),
}


def _copy_prefix(destination, destination_prefix, source, source_prefix):
    for key in destination:
        if key.startswith(destination_prefix):
            source_key = source_prefix + key[len(destination_prefix):]
            if source_key not in source:
                raise ValueError(f'Clean checkpoint is missing {source_key}')
            value = source[source_key]
            if value.shape != destination[key].shape:
                raise ValueError(f'Clean checkpoint shape mismatch: {source_key}')
            destination[key] = value


def load_clean_checkpoint(client, server, path, cfg):
    """Load weights on CPU, validate metadata, and return checkpoint metadata."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Clean checkpoint not found: {path}')
    checkpoint = torch.load(str(path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Expected a checkpoint dictionary: {path}')
    source_cfg = checkpoint.get('cfg', {})
    source_job = checkpoint.get('job', {})
    source_dataset = source_cfg.get('dataset', {}).get('name') or source_job.get('dataset_name')
    if source_dataset and source_dataset.lower() != cfg['dataset']['name'].lower():
        raise ValueError(f'Checkpoint dataset {source_dataset} != {cfg["dataset"]["name"]}')
    source_model = source_cfg.get('model', {})
    arch = cfg['model']['arch'].lower()
    source_arch = source_model.get('arch', arch).lower()
    if source_arch != arch and not {source_arch, arch}.issubset(RESNET18_NAMES):
        raise ValueError(f'Checkpoint architecture {source_arch} != {arch}')
    for field in ('num_classes', 'in_channels'):
        source_value = source_model.get(field, source_job.get(field))
        if source_value is not None and int(source_value) != int(cfg['model'][field]):
            raise ValueError(f'Checkpoint {field} does not match the experiment')

    if arch not in RESNET18_NAMES:
        if source_model.get('split', cfg['model']['split']) != cfg['model']['split']:
            raise ValueError('Automatic repartitioning is supported only for ResNet-18')
        client.load_state_dict(checkpoint['client'])
        server.load_state_dict(checkpoint['server'])
        return {'epoch': checkpoint.get('epoch', 0), 'source_split': source_model.get('split')}

    source_split = None
    if 'client' in checkpoint and 'server' in checkpoint:
        source_split = source_model.get('split')
        if source_split is None:
            stage_indices = {int(k.split('.')[1]) for k in checkpoint['server']
                             if k.startswith('layers.')}
            source_split = {3: 'early', 2: 'middle', 1: 'late'}.get(len(stage_indices))
        if source_split not in SERVER_STAGES:
            raise ValueError(f'Cannot identify checkpoint split: {path}')
        # Only stages actually executed by the source client are authoritative.
        last_client_stage = {'early': 1, 'middle': 2, 'late': 3}[source_split]
        prefixes = ('stem.',) + tuple(f'layer{i}.' for i in range(1, last_client_stage + 1))
        full = {k: v for k, v in checkpoint['client'].items() if k.startswith(prefixes)}
        for key, value in checkpoint['server'].items():
            if key.startswith('layers.'):
                _, index, suffix = key.split('.', 2)
                full[f'layer{SERVER_STAGES[source_split][int(index)]}.{suffix}'] = value
            elif key.startswith('fc.'):
                full[key] = value
    else:
        full = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        full = {(k[7:] if k.startswith('module.') else k): v for k, v in full.items()}

    client_state, server_state = client.state_dict(), server.state_dict()
    _copy_prefix(client_state, 'stem.', full, 'stem.')
    # The legacy client stores all three stages, even for shallow splits.
    for stage in (1, 2, 3):
        _copy_prefix(client_state, f'layer{stage}.', full, f'layer{stage}.')
    split = cfg['model']['split']
    for index, stage in enumerate(SERVER_STAGES[split]):
        _copy_prefix(server_state, f'layers.{index}.', full, f'layer{stage}.')
    _copy_prefix(server_state, 'fc.', full, 'fc.')
    client.load_state_dict(client_state)
    server.load_state_dict(server_state)
    return {'epoch': checkpoint.get('epoch', 0), 'source_split': source_split,
            'requested_split': split, 'path': str(path.resolve())}
