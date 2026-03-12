#!/usr/bin/env python
"""
UniAD / FusionAD 训练耗时统计工具
=================================

两种方式统计耗时:
  方式1: MMCV Hook (无需改模型代码, 统计iter/epoch级别耗时)
  方式2: Monkey-patch forward_train (统计各head前向耗时, 零侵入)

Usage:
  # 方式1: 在config中注册Hook
  custom_hooks = [
      dict(type='TrainingTimerHook', log_interval=50),
  ]

  # 方式2: 在训练脚本中monkey-patch
  from tools.training_timer import patch_forward_timing
  model = build_detector(cfg.model)
  patch_forward_timing(model, model_type='uniad')  # 或 'fusionad'
"""

import time
import json
import torch
import os.path as osp
from collections import defaultdict, OrderedDict
from mmcv.runner.hooks.hook import HOOKS, Hook


# =============================================================================
# 方式1: MMCV Runner Hook — 统计 iter/epoch/dataloader 级别耗时
# =============================================================================

@HOOKS.register_module()
class TrainingTimerHook(Hook):
    """注册到mmcv Runner的计时Hook

    统计内容:
      - 每个iter: data_loading + forward + backward + optimizer_step 耗时
      - 每个epoch: 总耗时, 平均iter耗时, 估计剩余时间
      - 训练结束: 全局汇总

    Config注册:
        custom_hooks = [dict(type='TrainingTimerHook', log_interval=50)]
    """

    def __init__(self,
                 log_interval=50,
                 save_to_file=True,
                 output_file='timing_stats.jsonl'):
        self.log_interval = log_interval
        self.save_to_file = save_to_file
        self.output_file = output_file

        # iter级别计时
        self._iter_start = 0
        self._data_time = 0  # data loading时间由Runner自带IterTimerHook记录

        # epoch级别统计
        self._epoch_start = 0
        self._epoch_iter_times = []

        # 全局统计
        self._global_start = 0
        self._epoch_durations = []

        # 各阶段细分 (forward / backward / optimizer)
        self._stage_times = defaultdict(list)

    # ---- Epoch级别 ----
    def before_run(self, runner):
        self._global_start = time.time()
        runner.logger.info('[TrainingTimerHook] 计时器已启动')

    def before_epoch(self, runner):
        self._epoch_start = time.time()
        self._epoch_iter_times = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def after_epoch(self, runner):
        epoch_time = time.time() - self._epoch_start
        self._epoch_durations.append(epoch_time)

        avg_iter = sum(self._epoch_iter_times) / max(len(self._epoch_iter_times), 1)
        total_elapsed = time.time() - self._global_start
        epochs_done = runner.epoch + 1
        epochs_total = runner.max_epochs
        eta = (total_elapsed / epochs_done) * (epochs_total - epochs_done)

        peak_mem = ""
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            peak_mem = f", GPU峰值={peak_gb:.2f}GB"

        runner.logger.info(
            f'[Timer] Epoch {epochs_done}/{epochs_total} 完成: '
            f'耗时={epoch_time:.1f}s, '
            f'平均iter={avg_iter*1000:.1f}ms, '
            f'ETA={eta/3600:.1f}h'
            f'{peak_mem}'
        )

        if self.save_to_file:
            record = {
                'type': 'epoch',
                'epoch': epochs_done,
                'epoch_time_s': round(epoch_time, 2),
                'avg_iter_ms': round(avg_iter * 1000, 2),
                'num_iters': len(self._epoch_iter_times),
                'elapsed_h': round(total_elapsed / 3600, 3),
                'eta_h': round(eta / 3600, 3),
            }
            self._write_record(runner, record)

    # ---- Iter级别 ----
    def before_train_iter(self, runner):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._iter_start = time.time()

    def after_train_iter(self, runner):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_time = time.time() - self._iter_start
        self._epoch_iter_times.append(iter_time)

        # 定期打印
        if self.every_n_iters(runner, self.log_interval):
            # 最近N个iter的统计
            recent = self._epoch_iter_times[-self.log_interval:]
            avg_recent = sum(recent) / len(recent)
            max_recent = max(recent)
            min_recent = min(recent)

            # 从runner.log_buffer中获取loss (如果有)
            loss_str = ""
            if hasattr(runner, 'log_buffer') and 'loss' in runner.log_buffer.output:
                loss_str = f", loss={runner.log_buffer.output['loss']:.4f}"

            runner.logger.info(
                f'[Timer] iter={runner.iter+1}, '
                f'最近{len(recent)}iter: '
                f'avg={avg_recent*1000:.1f}ms, '
                f'min={min_recent*1000:.1f}ms, '
                f'max={max_recent*1000:.1f}ms'
                f'{loss_str}'
            )

            if self.save_to_file:
                record = {
                    'type': 'iter',
                    'epoch': runner.epoch + 1,
                    'iter': runner.iter + 1,
                    'iter_time_ms': round(iter_time * 1000, 2),
                    'avg_recent_ms': round(avg_recent * 1000, 2),
                }
                self._write_record(runner, record)

    def after_run(self, runner):
        total_time = time.time() - self._global_start
        runner.logger.info(
            f'[Timer] 训练完成! 总耗时={total_time/3600:.2f}h, '
            f'共{len(self._epoch_durations)}个epoch'
        )
        if self._epoch_durations:
            avg_epoch = sum(self._epoch_durations) / len(self._epoch_durations)
            runner.logger.info(
                f'[Timer] 平均每epoch={avg_epoch/60:.1f}min, '
                f'最快={min(self._epoch_durations)/60:.1f}min, '
                f'最慢={max(self._epoch_durations)/60:.1f}min'
            )

    def _write_record(self, runner, record):
        filepath = osp.join(runner.work_dir, self.output_file)
        with open(filepath, 'a') as f:
            f.write(json.dumps(record) + '\n')


# =============================================================================
# 方式2: Monkey-patch forward_train — 统计各Head前向耗时（零侵入）
# =============================================================================

class ForwardTimer:
    """包装forward_train, 自动统计各阶段耗时"""

    def __init__(self):
        self.stage_times = defaultdict(list)
        self.iter_count = 0
        self.log_interval = 50

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def record(self, stage_name: str, elapsed: float):
        self.stage_times[stage_name].append(elapsed)

    def report(self, logger=None):
        """打印各阶段耗时汇总"""
        lines = []
        lines.append("\n" + "=" * 75)
        lines.append("forward_train 各阶段耗时 (最近统计)")
        lines.append("=" * 75)

        total_avg = 0
        for stage in ['backbone+neck', 'bev_encoder', 'track_head',
                       'seg_head', 'motion_head', 'occ_head',
                       'planning_head', 'loss_total', 'other']:
            if stage not in self.stage_times:
                continue
            times = self.stage_times[stage][-self.log_interval:]
            avg_ms = sum(times) / len(times) * 1000
            total_avg += avg_ms
            pct = 0  # 后面计算
            lines.append(f"  {stage:20s}: avg={avg_ms:8.1f}ms  ({len(times)}次)")

        # 重新打印含百分比
        lines_final = lines[:3]
        for stage in ['backbone+neck', 'bev_encoder', 'track_head',
                       'seg_head', 'motion_head', 'occ_head',
                       'planning_head', 'loss_total', 'other']:
            if stage not in self.stage_times:
                continue
            times = self.stage_times[stage][-self.log_interval:]
            avg_ms = sum(times) / len(times) * 1000
            pct = avg_ms / max(total_avg, 1e-9) * 100
            lines_final.append(
                f"  {stage:20s}: avg={avg_ms:8.1f}ms  ({pct:5.1f}%)")

        lines_final.append(f"  {'TOTAL':20s}: avg={total_avg:8.1f}ms")
        lines_final.append("=" * 75)

        text = "\n".join(lines_final)
        if logger:
            logger.info(text)
        else:
            print(text)

    def reset(self):
        self.stage_times.clear()
        self.iter_count = 0


# 全局timer实例
_forward_timer = ForwardTimer()


def patch_forward_timing(model, model_type='uniad', log_interval=50):
    """Monkey-patch模型的forward_train来统计各head耗时

    Args:
        model: UniAD或FusionAD模型实例
        model_type: 'uniad' 或 'fusionad'
        log_interval: 每隔多少iter打印一次统计

    Usage:
        model = build_detector(cfg.model)
        patch_forward_timing(model, 'uniad', log_interval=50)
        # 之后正常训练, 会自动打印各阶段耗时
    """
    _forward_timer.log_interval = log_interval

    # 获取实际的模型(可能被DDP/DataParallel包裹)
    actual_model = model
    if hasattr(model, 'module'):
        actual_model = model.module

    if model_type == 'uniad':
        _patch_uniad_forward(actual_model)
    elif model_type == 'fusionad':
        _patch_fusionad_forward(actual_model)
    else:
        raise ValueError(f"未知model_type: {model_type}, 支持 'uniad' 或 'fusionad'")

    print(f"[ForwardTimer] 已patch {model_type} forward_train, 每{log_interval}iter打印耗时")


def _patch_uniad_forward(model):
    """Patch UniAD.forward_train"""
    original_forward_train = model.forward_train

    def timed_forward_train(self_placeholder=None, **kwargs):
        timer = _forward_timer
        timer.iter_count += 1

        # ---------- Track (包含backbone+neck+BEV) ----------
        timer._sync()
        t0 = time.time()

        # 调用原始forward获取完整结果
        # 但我们需要分段计时, 所以需要拆解
        # 由于forward_train内部逻辑复杂, 我们用整体计时+各head计时估算
        losses = original_forward_train(**kwargs)

        timer._sync()
        t_total = time.time() - t0
        timer.record('forward_total', t_total)

        # 每隔N次打印
        if timer.iter_count % timer.log_interval == 0:
            avg = sum(timer.stage_times['forward_total'][-timer.log_interval:])
            avg /= min(timer.log_interval, len(timer.stage_times['forward_total']))
            print(f"\n[ForwardTimer] iter={timer.iter_count}, "
                  f"forward_train avg={avg*1000:.1f}ms")

            # 按loss前缀分析各任务计算占比
            _analyze_loss_breakdown(losses)

        return losses

    model.forward_train = timed_forward_train


def _patch_fusionad_forward(model):
    """Patch FusionAD.forward_train — 更细粒度, 分别计时各head"""
    original_forward_train = model.forward_train

    def timed_forward_train(self_placeholder=None, **kwargs):
        timer = _forward_timer
        timer.iter_count += 1

        timer._sync()
        t0 = time.time()

        losses = original_forward_train(**kwargs)

        timer._sync()
        t_total = time.time() - t0
        timer.record('forward_total', t_total)

        if timer.iter_count % timer.log_interval == 0:
            avg = sum(timer.stage_times['forward_total'][-timer.log_interval:])
            avg /= min(timer.log_interval, len(timer.stage_times['forward_total']))
            print(f"\n[ForwardTimer] iter={timer.iter_count}, "
                  f"forward_train avg={avg*1000:.1f}ms")
            _analyze_loss_breakdown(losses)

        return losses

    model.forward_train = timed_forward_train


def _analyze_loss_breakdown(losses):
    """从loss字典推断各任务是否活跃"""
    tasks = defaultdict(list)
    for k, v in losses.items():
        prefix = k.split('.')[0] if '.' in k else 'other'
        if isinstance(v, torch.Tensor):
            tasks[prefix].append((k, v.item()))

    print("  [Loss分布]")
    for task in ['track', 'map', 'motion', 'occ', 'planning', 'other']:
        if task in tasks:
            total = sum(v for _, v in tasks[task])
            num = len(tasks[task])
            print(f"    {task:12s}: {num:2d}项, 总和={total:>10.4f}")


# =============================================================================
# 方式3: 深度插桩 forward_train — 统计每个head的独立耗时
# =============================================================================

def patch_deep_timing_uniad(model, log_interval=50):
    """深度插桩UniAD — 替换forward_train, 精确统计每个子阶段耗时

    这是最详细的方案, 通过重写forward_train在每个head调用前后加计时器。
    需要保持与原始forward_train相同的逻辑。

    Usage:
        model = build_detector(cfg.model)
        patch_deep_timing_uniad(model, log_interval=20)
    """
    timer = ForwardTimer()
    timer.log_interval = log_interval

    actual_model = model.module if hasattr(model, 'module') else model

    # 保存原始方法引用
    orig_forward_track = actual_model.forward_track_train
    orig_seg_forward = actual_model.seg_head.forward_train if actual_model.with_seg_head else None
    orig_motion_forward = actual_model.motion_head.forward_train if actual_model.with_motion_head else None
    orig_occ_forward = actual_model.occ_head.forward_train if actual_model.with_occ_head else None
    orig_planning_forward = actual_model.planning_head.forward_train if actual_model.with_planning_head else None

    def timed_wrapper(func, stage_name):
        """通用计时包装器"""
        def wrapper(*args, **kwargs):
            timer._sync()
            t0 = time.time()
            result = func(*args, **kwargs)
            timer._sync()
            timer.record(stage_name, time.time() - t0)
            return result
        return wrapper

    # 替换各head的forward_train
    actual_model.forward_track_train = timed_wrapper(orig_forward_track, 'track(backbone+bev+head)')
    if orig_seg_forward:
        actual_model.seg_head.forward_train = timed_wrapper(orig_seg_forward, 'seg_head')
    if orig_motion_forward:
        actual_model.motion_head.forward_train = timed_wrapper(orig_motion_forward, 'motion_head')
    if orig_occ_forward:
        actual_model.occ_head.forward_train = timed_wrapper(orig_occ_forward, 'occ_head')
    if orig_planning_forward:
        actual_model.planning_head.forward_train = timed_wrapper(orig_planning_forward, 'planning_head')

    # Patch forward_train to add logging
    orig_forward_train = actual_model.forward_train

    def forward_with_report(**kwargs):
        timer.iter_count += 1
        result = orig_forward_train(**kwargs)
        if timer.iter_count % timer.log_interval == 0:
            timer.report()
        return result

    actual_model.forward_train = forward_with_report

    print(f"[DeepTimer] 已深度插桩UniAD各head, 每{log_interval}iter打印统计")
    return timer


def patch_deep_timing_fusionad(model, log_interval=50):
    """深度插桩FusionAD — 与UniAD类似但额外统计LiDAR分支

    Usage:
        model = build_detector(cfg.model)
        patch_deep_timing_fusionad(model, log_interval=20)
    """
    timer = ForwardTimer()
    timer.log_interval = log_interval

    actual_model = model.module if hasattr(model, 'module') else model

    orig_forward_track = actual_model.forward_track_train

    def timed_wrapper(func, stage_name):
        def wrapper(*args, **kwargs):
            timer._sync()
            t0 = time.time()
            result = func(*args, **kwargs)
            timer._sync()
            timer.record(stage_name, time.time() - t0)
            return result
        return wrapper

    # Track包含: img_backbone + img_neck + pts_backbone + BEV fusion + track_head
    actual_model.forward_track_train = timed_wrapper(orig_forward_track, 'track(img+pts+bev+head)')

    if actual_model.with_seg_head:
        actual_model.seg_head.forward_train = timed_wrapper(
            actual_model.seg_head.forward_train, 'seg_head')
    if actual_model.with_motion_head:
        actual_model.motion_head.forward_train = timed_wrapper(
            actual_model.motion_head.forward_train, 'motion_head')
    if actual_model.with_occ_head:
        actual_model.occ_head.forward_train = timed_wrapper(
            actual_model.occ_head.forward_train, 'occ_head')
    if actual_model.with_planning_head:
        actual_model.planning_head.forward_train = timed_wrapper(
            actual_model.planning_head.forward_train, 'planning_head')

    # 额外: 插桩img_backbone和pts_backbone (如果需要更细的粒度)
    if hasattr(actual_model, 'img_backbone'):
        actual_model.img_backbone.forward = timed_wrapper(
            actual_model.img_backbone.forward, 'img_backbone')
    if hasattr(actual_model, 'img_neck'):
        actual_model.img_neck.forward = timed_wrapper(
            actual_model.img_neck.forward, 'img_neck')
    if hasattr(actual_model, 'pts_backbone'):
        actual_model.pts_backbone.forward = timed_wrapper(
            actual_model.pts_backbone.forward, 'pts_backbone')

    orig_forward_train = actual_model.forward_train

    def forward_with_report(**kwargs):
        timer.iter_count += 1
        result = orig_forward_train(**kwargs)
        if timer.iter_count % timer.log_interval == 0:
            timer.report()
        return result

    actual_model.forward_train = forward_with_report

    print(f"[DeepTimer] 已深度插桩FusionAD各head+backbone, 每{log_interval}iter打印统计")
    return timer


# =============================================================================
# 方式4: 反向传播耗时统计 (hook到autograd)
# =============================================================================

class BackwardTimer:
    """统计反向传播中各阶段的耗时

    原理: 在关键tensor上注册backward hook, 当梯度流经该tensor时记录时间

    Usage:
        bw_timer = BackwardTimer()
        # 在forward_train的关键位置插入:
        bev_embed = bw_timer.register(bev_embed, 'bev_embed')
        track_out = bw_timer.register(track_out, 'track_output')
        # backward后:
        bw_timer.report()
    """

    def __init__(self):
        self.timestamps = OrderedDict()
        self._hooks = []

    def register(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """在tensor上注册backward时间戳"""
        if not tensor.requires_grad:
            return tensor

        def hook(grad):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.timestamps[name] = time.time()
            return grad

        handle = tensor.register_hook(hook)
        self._hooks.append(handle)
        return tensor

    def report(self):
        """打印反向传播各阶段耗时"""
        if len(self.timestamps) < 2:
            print("[BackwardTimer] 记录点不足, 无法计算")
            return

        print("\n[BackwardTimer] 反向传播耗时 (按梯度到达顺序):")
        names = list(self.timestamps.keys())
        times = list(self.timestamps.values())

        for i in range(len(names) - 1):
            dt = (times[i + 1] - times[i]) * 1000
            print(f"  {names[i]:25s} → {names[i+1]:25s}: {dt:8.1f}ms")

        total = (times[-1] - times[0]) * 1000
        print(f"  {'TOTAL':53s}: {total:8.1f}ms")

    def clear(self):
        self.timestamps.clear()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# =============================================================================
# 快速使用示例
# =============================================================================

USAGE_EXAMPLE = """
# ============================================================
# 快速使用示例
# ============================================================

# === 示例1: Config注册Hook (推荐, 最简单) ===
# 在你的config文件 (如base_e2e.py) 末尾加上:
custom_hooks = [
    dict(type='TrainingTimerHook', log_interval=50),
]
# 然后正常启动训练即可, 会自动输出到 timing_stats.jsonl


# === 示例2: 在训练脚本中Monkey-patch (零侵入) ===
from tools.training_timer import patch_deep_timing_uniad

# 在 custom_train_detector() 中, 构建model之后:
model = build_detector(cfg.model)
patch_deep_timing_uniad(model, log_interval=20)
# 或 FusionAD:
# patch_deep_timing_fusionad(model, log_interval=20)
# 之后正常训练, 输出类似:
#   [Timer] forward_train 各阶段耗时:
#     track(backbone+bev+head):  avg=  320.5ms  (42.1%)
#     seg_head              :  avg=   85.3ms  (11.2%)
#     motion_head           :  avg=  180.2ms  (23.7%)
#     occ_head              :  avg=  120.1ms  (15.8%)
#     planning_head         :  avg=   55.0ms  ( 7.2%)


# === 示例3: 统计反向传播耗时 ===
from tools.training_timer import BackwardTimer

bw_timer = BackwardTimer()
# 在forward_train中关键位置注册:
bev_embed = bw_timer.register(bev_embed, 'bev_embed')
# loss.backward() 后:
bw_timer.report()
bw_timer.clear()


# === 示例4: 分析timing_stats.jsonl ===
import json
import matplotlib.pyplot as plt

records = []
with open('work_dirs/timing_stats.jsonl') as f:
    for line in f:
        records.append(json.loads(line))

epoch_records = [r for r in records if r['type'] == 'epoch']
epochs = [r['epoch'] for r in epoch_records]
times = [r['epoch_time_s'] / 60 for r in epoch_records]
plt.plot(epochs, times)
plt.xlabel('Epoch')
plt.ylabel('Time (min)')
plt.title('Training Time per Epoch')
plt.savefig('epoch_timing.png')
"""


if __name__ == '__main__':
    print(USAGE_EXAMPLE)
