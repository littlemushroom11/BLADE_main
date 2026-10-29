# 表 3 / 表 5 补充四个随机种子

将整个 `BLADE_main` 文件夹复制到服务器，在这个文件夹中执行下列命令。
默认只跑 **seed 2、3、4、5**；原论文 seed 1 不会自动重跑。全部使用你提供的干净模型，
只改变缓存采样/数据增强、carrier 生成、投毒抽样等攻击阶段随机过程，不重新训练干净模型。
表 3 的代码范围是 **BLADE 行和攻击前 Clean model 行**；不包含 IBAS/SFI 的复现实验。

## 五个脚本

| 脚本 | 执行顺序 | 请求数量 | 与其他脚本重叠 |
|---|---|---:|---|
| `run_table3_cifar10_cifar100.py` | CIFAR-10 的四个种子，然后 CIFAR-100 的四个种子，均为 Split A | 8 | 两个数据集的 Split A |
| `run_table3_pathmnist_gtsrb.py` | PathMNIST 的四个种子，然后 GTSRB 的四个种子，均为 Split A | 8 | 无 |
| `run_table5_cifar10.py` | 每个种子依次 A、B、C | 12 | 与表 3 共用 4 个 A 实验 |
| `run_table5_cifar100.py` | 每个种子依次 A、B、C | 12 | 与表 3 共用 4 个 A 实验 |
| `run_table5_tinyimagenet.py` | 每个种子依次 A、B、C | 12 | 无 |

五个脚本共请求 52 次，去重后实际运行 **44 次**。无论先执行表 3 还是表 5，只要
`--output-dir`、数据路径、干净权重和其余实验参数一致，重复实验均会显示 `[REUSE]`。
`--device` 可以不同，GPU 分配不影响复用判断。
如果同时启动包含同一实验的脚本，操作系统文件锁会让它们依次检查/复用，不会重复训练同一实验。
锁文件可以保留，进程退出后锁由操作系统释放。

## 服务器命令

在你已有的 CUDA PyTorch 环境里执行（也可以用 `conda activate qe2`）。
三个路径均可由终端指定，支持包含空格的路径。以下变量只需设置一次：

```bash
cd /path/to/BLADE_main
DATA=/path/to/data
OUT=/path/to/outputs/table3_table5_multiseed
CLEAN=/path/to/clean_resnet18_bank

python run_table3_cifar10_cifar100.py --data-dir "$DATA" --output-dir "$OUT" --clean-checkpoint-dir "$CLEAN"
python run_table3_pathmnist_gtsrb.py --data-dir "$DATA" --output-dir "$OUT" --clean-checkpoint-dir "$CLEAN"
python run_table5_cifar10.py --data-dir "$DATA" --output-dir "$OUT" --clean-checkpoint-dir "$CLEAN"
python run_table5_cifar100.py --data-dir "$DATA" --output-dir "$OUT" --clean-checkpoint-dir "$CLEAN"
python run_table5_tinyimagenet.py --data-dir "$DATA" --output-dir "$OUT" --clean-checkpoint-dir "$CLEAN"
```

每个命令增加 `--dry-run` 可只查看计划，不创建输出、不加载数据、不训练。
脚本内部始终从自身所在目录启动 `BLADE.run_experiment`，因此也支持从别处通过绝对路径运行。

单卡不建议同时运行多个训练脚本；多卡可分别设置 `CUDA_VISIBLE_DEVICES=0/1/...`，
或用 `--device cuda:0`、`--device cuda:1`。默认 `cuda`，CUDA 不可用时会报错。

## 干净权重

推荐直接使用已有干净权重库：

```text
clean_resnet18_bank/
  cifar10/cifar10_resnet18_full_best.pt
  cifar100/cifar100_resnet18_full_best.pt
  pathmnist/pathmnist_resnet18_full_best.pt
  gtsrb/gtsrb_resnet18_full_best.pt
  tinyimagenet/tinyimagenet_resnet18_full_best.pt
```

同时支持平铺在根目录的同名文件，以及本仓库导出的
`{dataset}_resnet18_split_early_best.pt`、`..._middle_best.pt`、`..._late_best.pt`。
每个数据集优先选择完整权重，否则依次选择 early/middle/late；A/B/C 始终从同一份源权重加载。
加载器能在内存中重新切分本项目的 ResNet-18，无需额外导出 Split C 文件，也不修改输入权重。
完整权重支持本项目的 `model` 字典、`state_dict` 字典或直接的 state dict。
切分权重格式为 `client` / `server` 字典。其他实现的 ResNet-18 命名并不保证兼容。

自定义文件名时，可以覆盖具体数据集的权重路径：

```bash
python run_table3_cifar10_cifar100.py \
  --data-dir "$DATA" --output-dir "$OUT" \
  --cifar10-checkpoint /weights/c10.pt \
  --cifar100-checkpoint /weights/c100.pt

python run_table5_tinyimagenet.py \
  --data-dir "$DATA" --output-dir "$OUT" \
  --clean-checkpoint /weights/tiny.pt
```

同一数据集的表 3 / 表 5 请传相同源权重；不同序列化文件即使训练参数等价，其文件哈希也可能不同，
脚本会拒绝直接复用。脚本按配置、权重内容 SHA-256 和实现代码 SHA-256 校验历史结果。

## 数据与实验口径

- `--data-dir` 默认是公共数据根目录，其中包含 CIFAR 文件、`gtsrb/`、
  `pathmnist_64.npz` 和 `tiny-imagenet-200/`。PathMNIST 需要官方键名的 64×64 RGB NPZ。
- Tiny ImageNet 单独脚本也支持直接传 `tiny-imagenet-200` 目录。
- CIFAR/GTSRB 默认沿用 torchvision 的下载逻辑；离线服务器增加 `--no-download`。
- 模型为 **ResNet-18**。A/B/C = early/middle/late = layer1/layer2/layer3 后切分。
- 中心 4×4 checkerboard、目标类别 2、64 对 carriers、top-k 比例 10%、范数上限 8、
  所有类别按类抽取约 1% 特征投毒、仅微调 server，均沿用现有配置。
- **攻击 epoch 默认按论文正文：CIFAR-10、CIFAR-100、PathMNIST、GTSRB 为 10，
  Tiny ImageNet 为 20。** 原始 CIFAR-100、PathMNIST YAML 是 20，批量脚本会覆盖为 10；
  原始 YAML 保留，单实验入口仍按原 YAML 执行。需要旧口径时加 `--attack-epochs 20`，
  表 3 / 表 5 的同一数据集要一起修改，否则不能复用。
- 默认 batch size 256、workers 4，可用 `--batch-size`、`--num-workers` 修改；
  同一数据集的两个脚本必须一致。
- 默认 `--seeds 2 3 4 5`；也接受 `--seeds 2,3,4,5`。自定义种子时请保持不同脚本一致。
- ASR 主列使用 `input_asr_non_target`，排除原标签已是目标类别的样本；同时保留 `asr_all`。
- 汇总选取指定**最终 epoch**，不选最高 ASR epoch。准确率转换为百分数，标准差使用样本标准差 `ddof=1`。

论文的表 3 与表 5 中 CIFAR-100 Split A 的单次 BA 分别为 75.35 和 75.38，存在口径差异。
新实验会严格复用同一结果；seed 1 也应根据原始日志确认后使用同一个实验，不能把这两个数混作相同实验。

## 输出、断点重启与五种子汇总

所有脚本使用同一个输出根目录：

```text
OUT/
  runs/cifar10/resnet18/split_A/seed_2/
    config.yaml
    resolved_config.json
    console.log
    run_status.json
    metrics.csv
    ...
  manifests/table3_cifar10_cifar100.json
  summaries/table3_cifar10_cifar100/
    per_seed_results.csv
    mean_std.csv
    table.md
    summary.json
  summaries/table5_cifar10/
    ...
```

`tail -f "$OUT/runs/cifar10/resnet18/split_A/seed_2/console.log"` 可以查看训练日志。
原命令重复执行会验证并跳过已完成任务；未完成的单个实验会从干净权重重新开始，
后续任务继续执行。攻击特征缓存训练结束后仍按原逻辑清理，不保存攻击模型。
脚本失败时立即停止，并保存已经完成实验的汇总。更换配置、权重或实现代码后请使用新输出目录。

默认汇总只有 **4 个新增种子**，`n=4`、`complete_5seeds=false`，不会冒充五种子结果。
要合并原 seed 1，准备一个 JSON 文件（填**原始结果目录**，每个目录要有
`resolved_config.json` 和包含攻击前评估的 `metrics.csv`）：

```json
{
  "cifar10/A": "/old/seed1/cifar10_A",
  "cifar10/B": "/old/seed1/cifar10_B",
  "cifar10/C": "/old/seed1/cifar10_C",
  "cifar100/A": "/old/seed1/cifar100_A",
  "cifar100/B": "/old/seed1/cifar100_B",
  "cifar100/C": "/old/seed1/cifar100_C",
  "pathmnist/A": "/old/seed1/pathmnist_A",
  "gtsrb/A": "/old/seed1/gtsrb_A",
  "tinyimagenet/A": "/old/seed1/tiny_A",
  "tinyimagenet/B": "/old/seed1/tiny_B",
  "tinyimagenet/C": "/old/seed1/tiny_C"
}
```

只填已有项即可；相对路径相对于 JSON 文件。**显式导入表示你确认这些历史实验使用的是
与当前对应数据集相同的干净模型**；旧日志没有权重 SHA-256，无法追溯验证历史权重内容。
脚本会核对 seed、模型、切分、预处理参数、batch size 和攻击参数，并读取一致的指定 epoch。
旧日志如果跑到了第 20 轮而本次设置为 10，会读取旧日志**第 10 轮**，不会取第 20 轮冒充。
原目录只读，绝不重写。

给上述任意命令增加 `--seed1-results /path/to/seed1_results.json` 即可合并，
也可以实验完成后增加 `--aggregate-only` 只重建汇总：

```bash
python run_table5_cifar10.py \
  --data-dir "$DATA" --output-dir "$OUT" --clean-checkpoint-dir "$CLEAN" \
  --seed1-results /path/to/seed1_results.json --aggregate-only
```

五个脚本各自生成自己的汇总，完成后可对每个命令执行一次 `--aggregate-only`。
只有导入成功且五个不同种子齐全时，该设置才标记为五种子完成。
若没有原始 seed 1 日志，也可显式 `--seeds 1 2 3 4 5` 重跑五种子；已有的 2–5 会复用。

## 原单实验入口的新增参数

无需修改 YAML 也可以执行单次实验：

```bash
python -m BLADE.run_experiment \
  --config configs/cifar10_resnet18_layer1_checkerboard_center_4x4_ratio10_carrier64_attack10_multisource_all_1pct_bs256.yaml \
  --data-dir /path/to/data \
  --output-dir /path/to/output/single_run \
  --clean-checkpoint /path/to/cifar10_resnet18_full_best.pt \
  --seed 2 --split early --device cuda --num-workers 4
```

## 代码验证（不运行训练）

```bash
python -m unittest discover -s tests -v
```

覆盖去重后的 44 个实验、数据集顺序、跨表配置一致、错误结果拒绝复用、失败重启、
最终 epoch 汇总、seed 1 导入，以及完整/各切分权重重新切分后的前向输出一致性。
