"""Shared runner for Table 3 / Table 5 additional attack seeds.

No clean-model training is performed. Separate entry scripts share canonical
run directories and operating-system locks, including when launched together.
"""

import argparse
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'cifar10': 'cifar10_resnet18_layer1_checkerboard_center_4x4_ratio10_carrier64_attack10_multisource_all_1pct_bs256.yaml',
    'cifar100': 'cifar100_resnet18_layer1_checkerboard_center_4x4_ratio10_carrier64_attack10.yaml',
    'pathmnist': 'pathmnist_resnet18_layer1_checkerboard_center_4x4_ratio10_carrier64_attack10.yaml',
    'gtsrb': 'GTSRB_resnet18_layer1_checkerboard_center_4x4_ratio10_carrier64_attack10.yaml',
    'tinyimagenet': 'tinyimagenet_resnet18_layer1_checkerboard_center_4x4_ratio10_carrier64_attack10.yaml',
}
SPLITS = {'A': 'early', 'B': 'middle', 'C': 'late'}
SUITES = {
    'table3_cifar10_cifar100': (('cifar10', 'cifar100'), ('A',)),
    'table3_pathmnist_gtsrb': (('pathmnist', 'gtsrb'), ('A',)),
    'table5_cifar10': (('cifar10',), ('A', 'B', 'C')),
    'table5_cifar100': (('cifar100',), ('A', 'B', 'C')),
    'table5_tinyimagenet': (('tinyimagenet',), ('A', 'B', 'C')),
}
PERCENT_METRICS = ('base_acc', 'clean_model_asr', 'ba', 'asr', 'asr_all', 'latent_asr', 'clean_drop_pp')
METRICS = PERCENT_METRICS + ('bridge_cos',)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def implementation_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'BLADE').glob('*.py')):
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_hash(cfg):
    value = copy.deepcopy(cfg)
    value.pop('output_dir', None)
    value.pop('device', None)  # The same result can be reused across GPU assignments.
    # Checkpoint contents identify the clean model, even if its path changes.
    if 'reproducibility' in value:
        value.pop('warmup_checkpoint', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()


def resolve_path(value):
    return Path(os.path.normpath(str(Path(value).expanduser().resolve())))


def resolve_checkpoint(args, dataset):
    override = getattr(args, dataset + '_checkpoint') or args.clean_checkpoint
    if override:
        return resolve_path(override)
    if args.clean_checkpoint_dir is None:
        raise ValueError('Specify --clean-checkpoint-dir or a checkpoint file override')
    root = resolve_path(args.clean_checkpoint_dir)
    # Always choose the same source checkpoint for all splits.
    names = [f'{dataset}_resnet18_full_best.pt'] + [
        f'{dataset}_resnet18_split_{split}_best.pt' for split in SPLITS.values()]
    for directory in (root / dataset, root):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    if args.dry_run:
        return root / dataset / names[0]
    raise FileNotFoundError(f'No clean ResNet-18 checkpoint for {dataset} under {root}')


def parse_seeds(values):
    seeds = [int(part) for value in values for part in value.split(',')]
    if not seeds or any(seed < 0 or seed >= 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be distinct integers in [0, 2**32)')
    return seeds


def build_jobs(suite, args):
    datasets, splits = SUITES[suite]
    seeds = parse_seeds(args.seeds)
    if args.clean_checkpoint and len(datasets) != 1:
        raise ValueError('For a two-dataset script use --clean-checkpoint-dir or per-dataset overrides')
    output_root = resolve_path(args.output_dir)
    code_hash = implementation_hash()
    jobs = []
    for dataset in datasets:
        checkpoint = resolve_checkpoint(args, dataset)
        if not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError(f'Clean checkpoint not found: {checkpoint}')
        checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else 'dry-run-only'
        base = yaml.safe_load((ROOT / 'configs' / CONFIGS[dataset]).read_text(encoding='utf-8'))
        for seed in seeds:
            for split in splits:
                cfg = copy.deepcopy(base)
                run_dir = output_root / 'runs' / dataset / 'resnet18' / f'split_{split}' / f'seed_{seed}'
                cfg['output_dir'] = str(run_dir)
                cfg['seed'] = seed
                cfg['device'] = args.device
                cfg['dataset']['data_dir'] = str(resolve_path(args.data_dir))
                cfg['dataset']['num_workers'] = args.num_workers
                cfg['dataset']['download'] = not args.no_download
                if args.batch_size is not None:
                    cfg['dataset']['batch_size'] = args.batch_size
                cfg['model']['split'] = SPLITS[split]
                cfg['warmup_checkpoint']['load_path'] = str(checkpoint)
                cfg['train']['attack_epochs'] = args.attack_epochs or (20 if dataset == 'tinyimagenet' else 10)
                cfg['reproducibility'] = {
                    'clean_checkpoint_sha256': checkpoint_hash,
                    'implementation_sha256': code_hash,
                    'protocol': 'table3_table5_v1',
                }
                jobs.append({'dataset': dataset, 'split': split, 'seed': seed,
                             'job_id': f'{dataset}_resnet18_split{split}_seed{seed}',
                             'config': cfg, 'config_sha256': config_hash(cfg)})
    return jobs


@contextmanager
def run_lock(path):
    """Locks are released by the OS on exit/crash; the file itself may remain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(2)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def result_from_metrics(job, run_dir, source, allow_later_epochs=False):
    with (Path(run_dir) / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    pre = [row for row in rows if row['stage'] == 'pre_attack_eval']
    attack = [row for row in rows if row['stage'] == 'cached_feature_attack']
    epochs = int(job['config']['train']['attack_epochs'])
    actual_epochs = [int(float(row['epoch'])) for row in attack]
    expected = list(range(1, epochs + 1))
    valid = actual_epochs[:epochs] == expected if allow_later_epochs else actual_epochs == expected
    if len(pre) != 1 or not valid:
        raise ValueError(f'Incomplete/duplicate attack metrics in {run_dir}; expected epochs 1..{epochs}')
    final = attack[epochs - 1]
    def metric(row, name):
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError(f'Non-finite {name} in {run_dir}')
        if name != 'bridge_cos' and not 0 <= value <= 1:
            raise ValueError(f'Accuracy outside [0,1]: {name} in {run_dir}')
        return value
    base_acc = 100 * metric(pre[0], 'clean_acc')
    ba = 100 * metric(final, 'clean_acc')
    return {
        'dataset': job['dataset'], 'split': job['split'], 'seed': job['seed'],
        'epoch': epochs, 'base_acc': base_acc,
        'clean_model_asr': 100 * metric(pre[0], 'input_asr_non_target'),
        'ba': ba, 'asr': 100 * metric(final, 'input_asr_non_target'),
        'asr_all': 100 * metric(final, 'input_asr_all'),
        'latent_asr': 100 * metric(final, 'latent_asr'),
        'clean_drop_pp': base_acc - ba, 'bridge_cos': metric(final, 'bridge_cos'),
        'source': source, 'run_dir': str(run_dir),
    }


def load_result(job):
    run_dir = Path(job['config']['output_dir'])
    status_path = run_dir / 'run_status.json'
    if not status_path.is_file():
        return None
    status = read_json(status_path)
    if status.get('config_sha256') != job['config_sha256']:
        raise ValueError(f'Existing run has different settings/weights/code: {run_dir}. Use a new --output-dir.')
    if status.get('state') != 'completed' or status.get('returncode') != 0:
        return None
    if config_hash(read_json(run_dir / 'resolved_config.json')) != job['config_sha256']:
        raise ValueError(f'Resolved configuration mismatch: {run_dir}')
    return result_from_metrics(job, run_dir, 'run')


def run_job(job):
    run_dir = Path(job['config']['output_dir'])
    print(f'[LOCK] {job["job_id"]}: waiting for exclusive run access', flush=True)
    with run_lock(run_dir.parent / f'.seed_{job["seed"]}.lock'):
        if load_result(job) is not None:
            print(f'[REUSE] {job["job_id"]}', flush=True)
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / 'config.yaml'
        if config_path.exists() and config_hash(yaml.safe_load(config_path.read_text(encoding='utf-8'))) != job['config_sha256']:
            raise ValueError(f'Existing unfinished run has different settings: {run_dir}. Use a new --output-dir.')
        config_path.write_text(yaml.safe_dump(job['config'], sort_keys=False), encoding='utf-8')
        status = {'state': 'running', 'started_at': now(), 'config_sha256': job['config_sha256']}
        write_json(run_dir / 'run_status.json', status)
        started = time.monotonic()
        command = [sys.executable, '-u', '-m', 'BLADE.run_experiment', '--config', str(config_path)]
        print(f'[RUN] {job["job_id"]}; log: {run_dir / "console.log"}', flush=True)
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        try:
            with (run_dir / 'console.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(command, cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT)
            status.update(returncode=process.returncode, finished_at=now(), wall_seconds=time.monotonic() - started)
            if process.returncode:
                raise RuntimeError(f'Experiment failed with exit code {process.returncode}; see {run_dir / "console.log"}')
            if config_hash(read_json(run_dir / 'resolved_config.json')) != job['config_sha256']:
                raise ValueError(f'Experiment changed its requested configuration: {run_dir}')
            result_from_metrics(job, run_dir, 'run')
            status['state'] = 'completed'
        except BaseException as exc:
            status.update(state='failed', error=str(exc), finished_at=now(), wall_seconds=time.monotonic() - started)
            write_json(run_dir / 'run_status.json', status)
            raise
        write_json(run_dir / 'run_status.json', status)
        print(f'[DONE] {job["job_id"]}', flush=True)


def seed1_behavior(cfg):
    cfg = copy.deepcopy(cfg)
    for key in ('output_dir', 'device', 'warmup_checkpoint', 'reproducibility'):
        cfg.pop(key, None)
    for key in ('data_dir', 'num_workers', 'download'):
        cfg['dataset'].pop(key, None)
    cfg['train'].pop('attack_epochs', None)
    return cfg


def import_seed1(jobs, mapping_path):
    """Explicitly selected historical runs; input directories are never modified."""
    if mapping_path is None:
        return []
    mapping_path = resolve_path(mapping_path)
    mapping = read_json(mapping_path)
    records = []
    settings = {(job['dataset'], job['split']): job for job in jobs}
    for (dataset, split), original in settings.items():
        key = f'{dataset}/{split}'
        if key not in mapping:
            continue
        if any(job['seed'] == 1 and job['dataset'] == dataset and job['split'] == split for job in jobs):
            raise ValueError('Do not combine --seed1-results with --seeds containing 1')
        directory = Path(mapping[key]).expanduser()
        directory = (mapping_path.parent / directory).resolve() if not directory.is_absolute() else directory.resolve()
        cfg = read_json(directory / 'resolved_config.json')
        job = copy.deepcopy(original)
        job['seed'] = job['config']['seed'] = 1
        if seed1_behavior(cfg) != seed1_behavior(job['config']):
            raise ValueError(f'Seed-1 experiment settings differ for {key}: {directory}')
        records.append(result_from_metrics(job, directory, 'imported_seed1', allow_later_epochs=True))
    return records


def write_csv(path, records, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def aggregate(suite, jobs, args):
    records = [result for job in jobs for result in [load_result(job)] if result is not None]
    records.extend(import_seed1(jobs, args.seed1_results))
    records.sort(key=lambda r: (r['dataset'], r['split'], r['seed']))
    output = resolve_path(args.output_dir) / 'summaries' / suite
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'per_seed_results.csv', records,
              ['dataset', 'split', 'seed', 'epoch'] + list(METRICS) + ['source', 'run_dir'])
    summary = []
    for dataset in SUITES[suite][0]:
        for split in SUITES[suite][1]:
            items = [r for r in records if r['dataset'] == dataset and r['split'] == split]
            row = {'dataset': dataset, 'split': split, 'n': len(items),
                   'seeds': ','.join(str(r['seed']) for r in items), 'complete_5seeds': len(items) == 5}
            for metric in METRICS:
                values = [r[metric] for r in items]
                row[metric + '_mean'] = statistics.mean(values) if values else None
                row[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(row)
    write_csv(output / 'mean_std.csv', summary, list(summary[0]))
    write_json(output / 'summary.json', {
        'suite': suite, 'selection': 'fixed final epoch', 'asr_column': 'input_asr_non_target',
        'accuracy_unit': 'percent', 'std_ddof': 1, 'new_seeds': parse_seeds(args.seeds),
        'seed1_results': args.seed1_results, 'completed_requested_runs': sum(r['source'] == 'run' for r in records),
        'requested_runs': len(jobs), 'summary': summary,
    })
    lines = ['| Dataset | Split | n | Seeds | Base Acc. | BA | ASR |',
             '|---|---|---:|---|---:|---:|---:|']
    def cell(row, metric):
        mean, std = row[metric + '_mean'], row[metric + '_std']
        if mean is None:
            return 'pending'
        return f'{mean:.2f} +/- {std:.2f}' if std is not None else f'{mean:.2f}'
    for row in summary:
        lines.append(f'| {row["dataset"]} | {row["split"]} | {row["n"]} | {row["seeds"]} | '
                     f'{cell(row, "base_acc")} | {cell(row, "ba")} | {cell(row, "asr")} |')
    lines.extend(['', 'Sample standard deviation (ddof=1). Accuracy values are percentages.',
                  'n=4 means only the four additional seeds; it is not a five-seed result.',
                  'Seed-1 imports are explicitly supplied historical runs using the same clean model.'])
    (output / 'table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return records


def make_parser(suite):
    parser = argparse.ArgumentParser(description=f'{suite}: additional BLADE seeds; no clean-model training')
    parser.add_argument('--data-dir', required=True, help='Dataset root (or PathMNIST .npz / Tiny ImageNet directory).')
    parser.add_argument('--output-dir', required=True, help='Use the SAME directory for all five scripts to share runs.')
    parser.add_argument('--clean-checkpoint-dir', help='Clean bank root; supports dataset/ and flat layouts.')
    parser.add_argument('--clean-checkpoint', help='Checkpoint file for a single-dataset script.')
    for dataset in CONFIGS:
        parser.add_argument(f'--{dataset}-checkpoint', help=f'Override the checkpoint file for {dataset}.')
    parser.add_argument('--seeds', nargs='+', default=['2', '3', '4', '5'])
    parser.add_argument('--device', default='cuda', help='cpu, cuda, or cuda:N')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--attack-epochs', type=int, default=None, help='Default: 10, or 20 for Tiny ImageNet.')
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--seed1-results', help='JSON mapping dataset/A|B|C to historical seed-1 run directories.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print the plan without creating files/loading data.')
    mode.add_argument('--aggregate-only', action='store_true', help='Rebuild reports without training.')
    return parser


def main(suite, argv=None):
    args = make_parser(suite).parse_args(argv)
    if args.num_workers < 0 or (args.batch_size is not None and args.batch_size <= 0) or (args.attack_epochs is not None and args.attack_epochs <= 0):
        raise ValueError('num-workers must be nonnegative; batch-size/attack-epochs must be positive')
    jobs = build_jobs(suite, args)
    print(f'{suite}: {len(jobs)} requested runs; seeds={parse_seeds(args.seeds)}', flush=True)
    for job in jobs:
        cfg = job['config']
        print(f'  {job["job_id"]}: epochs={cfg["train"]["attack_epochs"]}, '
              f'checkpoint={cfg["warmup_checkpoint"]["load_path"]}, output={cfg["output_dir"]}', flush=True)
    if args.dry_run:
        return
    # Validate all explicitly imported historical results before running jobs.
    import_seed1(jobs, args.seed1_results)
    output_root = resolve_path(args.output_dir)
    write_json(output_root / 'manifests' / f'{suite}.json', {'suite': suite, 'jobs': jobs})
    if not args.aggregate_only:
        for index, job in enumerate(jobs, 1):
            print(f'[{index}/{len(jobs)}]', flush=True)
            try:
                run_job(job)
            finally:
                aggregate(suite, jobs, args)
    records = aggregate(suite, jobs, args)
    completed = sum(r['source'] == 'run' for r in records)
    print(f'Reports: {output_root / "summaries" / suite}; completed {completed}/{len(jobs)} requested runs.', flush=True)


if __name__ == '__main__':
    raise SystemExit('Use one of the five run_table*.py entry scripts.')
