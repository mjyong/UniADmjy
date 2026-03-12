#!/usr/bin/env python
"""
=============================================================================
UniAD & FusionAD 训练流程分析 + Debug工具代码
=============================================================================

## 1. 训练流程对比
====================

### UniAD 两阶段训练:
Stage 1 (Track+Map):  6 epochs, lr=2e-4
  - 训练: BEVFormerTrackHead + PansegformerHead
  - 冻结: img_backbone(ResNet101), img_neck解冻, BN解冻
  - 加载: bevformer_r101_dcn_24ep.pth (BEVFormer预训练)
  - queue_length=5 (5帧时序)

Stage 2 (E2E全任务): 20 epochs, lr=2e-4
  - 训练: MotionHead + OccHead + PlanningHead (新增模块)
  - 冻结: img_backbone + img_neck + BN + BEV encoder (Stage1已训好)
  - 加载: uniad_base_track_map.pth (Stage1 checkpoint)
  - queue_length=3 (省显存)

### FusionAD 训练 (LiDAR+Camera融合):
Stage 2 (E2E全任务): 36 epochs, lr=1e-4
  - 额外冻结控制(forward级别): freeze_track=True, freeze_seg=True, freeze_motion=True, freeze_occ=False
  - 冻结: img_backbone + img_neck(freeze_img_modules), BEV encoder
  - 额外有: LiDAR分支(SparseEncoderHD), PtsCrossAttention融合
  - queue_length=3

## 2. 冻结机制对比
====================

UniAD冻结方式:
  - 参数级: requires_grad=False (backbone/neck)
  - 模块级: freeze_bev_encoder → torch.no_grad() 包裹forward

FusionAD冻结方式(更灵活):
  - 参数级: freeze_img_modules → eval() + requires_grad=False
  - 前向级: freeze_track/seg/motion/occ → torch.no_grad() 包裹各head的forward
  - 只有非冻结模块的loss会被计算和回传

## 3. 加速训练策略
====================

策略1: 减少queue_length (5→3→1), 减少时序帧数
策略2: 冻结更多模块, 只训练目标head
策略3: 减小BEV分辨率 (200x200 → 100x100)
策略4: 使用FP16混合精度 (已内置auto_fp16)
策略5: 增大梯度累积 (减少通信开销)
策略6: 使用更小的backbone (ResNet50替代ResNet101)
策略7: 减少transformer层数
策略8: 分布式训练 + 数据并行
"""

import torch
import torch.nn as nn
import sys
import os
import json
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Part 1: 模型参数分析工具
# =============================================================================

def analyze_model_params(model: nn.Module, print_details: bool = True) -> Dict:
    """分析模型各模块的参数量和训练状态

    Usage:
        model = build_model(cfg)
        stats = analyze_model_params(model)
    """
    stats = {}
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    # 按顶层模块分组统计
    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_frozen = module_total - module_trainable

        stats[name] = {
            'total': module_total,
            'trainable': module_trainable,
            'frozen': module_frozen,
            'pct_trainable': f"{module_trainable/max(module_total,1)*100:.1f}%",
            'training_mode': 'train' if module.training else 'eval',
        }
        total_params += module_total
        trainable_params += module_trainable
        frozen_params += module_frozen

    stats['__summary__'] = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_pct': f"{trainable_params/max(total_params,1)*100:.1f}%",
        'total_MB': f"{total_params * 4 / 1024**2:.1f} MB",
        'trainable_MB': f"{trainable_params * 4 / 1024**2:.1f} MB",
    }

    if print_details:
        print("=" * 80)
        print("模型参数分析")
        print("=" * 80)
        for name, info in stats.items():
            if name == '__summary__':
                continue
            print(f"\n[{name}]")
            print(f"  总参数: {info['total']:>12,d}  |  可训练: {info['trainable']:>12,d}  |  "
                  f"冻结: {info['frozen']:>12,d}  |  可训练比例: {info['pct_trainable']}")
            print(f"  训练模式: {info['training_mode']}")

        summary = stats['__summary__']
        print("\n" + "=" * 80)
        print(f"总计: {summary['total_params']:,d} 参数 ({summary['total_MB']})")
        print(f"可训练: {summary['trainable_params']:,d} ({summary['trainable_pct']})")
        print(f"冻结: {summary['frozen_params']:,d}")
        print("=" * 80)

    return stats


def check_gradient_flow(model: nn.Module, loss: torch.Tensor) -> Dict:
    """检查梯度流是否正常 - 在loss.backward()后调用

    Usage:
        loss = model.forward_train(...)
        total_loss = sum(loss.values())
        total_loss.backward()
        grad_info = check_gradient_flow(model, total_loss)
    """
    grad_info = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_info[name] = {
                    'grad_norm': grad_norm,
                    'has_grad': True,
                    'is_zero': grad_norm < 1e-10,
                    'is_nan': torch.isnan(param.grad).any().item(),
                    'is_inf': torch.isinf(param.grad).any().item(),
                    'param_norm': param.data.norm(2).item(),
                }
            else:
                grad_info[name] = {
                    'grad_norm': 0.0,
                    'has_grad': False,
                    'is_zero': True,
                    'is_nan': False,
                    'is_inf': False,
                    'param_norm': param.data.norm(2).item(),
                }

    # 汇总
    no_grad_params = [k for k, v in grad_info.items() if not v['has_grad']]
    zero_grad_params = [k for k, v in grad_info.items() if v['is_zero'] and v['has_grad']]
    nan_params = [k for k, v in grad_info.items() if v['is_nan']]
    inf_params = [k for k, v in grad_info.items() if v['is_inf']]

    print("\n" + "=" * 80)
    print("梯度流检查")
    print("=" * 80)

    if nan_params:
        print(f"\n[WARNING] NaN梯度 ({len(nan_params)}个参数):")
        for p in nan_params[:10]:
            print(f"  - {p}")

    if inf_params:
        print(f"\n[WARNING] Inf梯度 ({len(inf_params)}个参数):")
        for p in inf_params[:10]:
            print(f"  - {p}")

    if no_grad_params:
        print(f"\n[INFO] 无梯度(requires_grad=True但grad=None) ({len(no_grad_params)}个参数):")
        for p in no_grad_params[:10]:
            print(f"  - {p}")
        if len(no_grad_params) > 10:
            print(f"  ... 还有{len(no_grad_params)-10}个")

    if zero_grad_params:
        print(f"\n[INFO] 梯度为零 ({len(zero_grad_params)}个参数):")
        for p in zero_grad_params[:10]:
            print(f"  - {p}")

    total_with_grad = sum(1 for v in grad_info.values() if v['has_grad'] and not v['is_zero'])
    print(f"\n总计: {len(grad_info)}个可训练参数, {total_with_grad}个有非零梯度")

    return grad_info


# =============================================================================
# Part 2: 冻结/解冻工具
# =============================================================================

def freeze_module(module: nn.Module, freeze_bn: bool = True):
    """冻结模块的所有参数"""
    for param in module.parameters():
        param.requires_grad = False
    if freeze_bn:
        module.eval()
    print(f"[FREEZE] 已冻结 {sum(1 for _ in module.parameters())} 个参数")


def unfreeze_module(module: nn.Module):
    """解冻模块的所有参数"""
    for param in module.parameters():
        param.requires_grad = True
    module.train()
    print(f"[UNFREEZE] 已解冻 {sum(1 for _ in module.parameters())} 个参数")


def selective_freeze_uniad(model, stage='stage2'):
    """UniAD选择性冻结

    Stage1: 训练track+map, 冻结backbone
    Stage2: 训练motion+occ+planning, 冻结backbone+neck+BEV+track+map
    """
    if stage == 'stage1':
        # Stage 1: 冻结backbone, 训练neck+BEV+track+map
        freeze_module(model.img_backbone, freeze_bn=True)
        # neck和BEV保持可训练
        print("[Stage1] 冻结img_backbone, 其余可训练")

    elif stage == 'stage2':
        # Stage 2: 冻结backbone+neck+BEV, 训练下游头
        freeze_module(model.img_backbone, freeze_bn=True)
        freeze_module(model.img_neck, freeze_bn=True)
        # BEV encoder通过forward中的torch.no_grad()冻结
        model.freeze_bev_encoder = True
        print("[Stage2] 冻结img_backbone, img_neck, BEV encoder")
        print("[Stage2] 可训练: motion_head, occ_head, planning_head, seg_head")

    elif stage == 'planning_only':
        # 只训练planning head (极速微调)
        for name, param in model.named_parameters():
            param.requires_grad = False
        for param in model.planning_head.parameters():
            param.requires_grad = True
        print("[Planning Only] 仅planning_head可训练")

    analyze_model_params(model)


def selective_freeze_fusionad(model,
                               freeze_track=True,
                               freeze_seg=True,
                               freeze_motion=True,
                               freeze_occ=False):
    """FusionAD选择性冻结 (更灵活的前向级冻结)

    FusionAD通过torch.no_grad()在forward中控制梯度流,
    冻结的模块仍然前向传播但不产生梯度。
    """
    model.freeze_track = freeze_track
    model.freeze_seg = freeze_seg
    model.freeze_motion = freeze_motion
    model.freeze_occ = freeze_occ

    print(f"[FusionAD冻结配置]")
    print(f"  Track:  {'冻结' if freeze_track else '可训练'}")
    print(f"  Seg:    {'冻结' if freeze_seg else '可训练'}")
    print(f"  Motion: {'冻结' if freeze_motion else '可训练'}")
    print(f"  Occ:    {'冻结' if freeze_occ else '可训练'}")

    # 注意: FusionAD的冻结是forward级别的, 参数仍然requires_grad=True
    # 但torch.no_grad()阻止了梯度计算, 所以优化器不会更新这些参数
    # 如需完全冻结(省显存), 可以额外设置requires_grad=False:
    if freeze_track and hasattr(model, 'pts_bbox_head'):
        for param in model.pts_bbox_head.parameters():
            param.requires_grad = False
        print("  [额外] pts_bbox_head参数已设为requires_grad=False (省显存)")

    if freeze_seg and hasattr(model, 'seg_head'):
        for param in model.seg_head.parameters():
            param.requires_grad = False
        print("  [额外] seg_head参数已设为requires_grad=False (省显存)")

    if freeze_motion and hasattr(model, 'motion_head'):
        for param in model.motion_head.parameters():
            param.requires_grad = False
        print("  [额外] motion_head参数已设为requires_grad=False (省显存)")


# =============================================================================
# Part 3: 训练Debug Hook
# =============================================================================

class TrainingDebugHook:
    """训练过程Debug钩子 - 注入到mmcv runner中

    Usage:
        hook = TrainingDebugHook(log_interval=10, check_grad=True)
        runner.register_hook(hook)

    或者手动在训练循环中调用:
        hook = TrainingDebugHook()
        for iter, data in enumerate(dataloader):
            losses = model.forward_train(**data)
            hook.after_train_iter_manual(model, losses, iter)
    """

    def __init__(self,
                 log_interval: int = 10,
                 check_grad: bool = True,
                 check_loss_nan: bool = True,
                 monitor_memory: bool = True,
                 save_loss_history: bool = True,
                 loss_log_file: str = 'training_losses.jsonl'):
        self.log_interval = log_interval
        self.check_grad = check_grad
        self.check_loss_nan = check_loss_nan
        self.monitor_memory = monitor_memory
        self.save_loss_history = save_loss_history
        self.loss_log_file = loss_log_file
        self.loss_history = []
        self.iter_count = 0
        self.start_time = time.time()

    def after_train_iter_manual(self, model, losses: Dict, iteration: int):
        """手动调用版本 - 在每个训练迭代后调用"""
        self.iter_count = iteration

        if iteration % self.log_interval != 0:
            return

        print(f"\n{'='*60}")
        print(f"[DEBUG] Iteration {iteration}")
        print(f"{'='*60}")

        # 1. 检查loss
        self._check_losses(losses)

        # 2. 检查显存
        if self.monitor_memory and torch.cuda.is_available():
            self._check_memory()

        # 3. 检查梯度
        if self.check_grad:
            self._check_gradients_summary(model)

        # 4. 保存loss历史
        if self.save_loss_history:
            self._save_loss(losses, iteration)

    def _check_losses(self, losses: Dict):
        """检查各任务loss"""
        print("\n[Loss值]")
        task_losses = {}
        for k, v in sorted(losses.items()):
            val = v.item() if isinstance(v, torch.Tensor) else v
            # 按任务分组
            task = k.split('.')[0] if '.' in k else 'other'
            if task not in task_losses:
                task_losses[task] = 0.0
            task_losses[task] += val

            # 检查异常值
            flag = ""
            if isinstance(val, float):
                if val != val:  # NaN
                    flag = " [NaN!]"
                elif abs(val) > 1e6:
                    flag = " [极大值!]"
                elif abs(val) == float('inf'):
                    flag = " [Inf!]"
            print(f"  {k:40s} = {val:>12.6f}{flag}")

        print("\n[按任务汇总]")
        for task, total in sorted(task_losses.items()):
            print(f"  {task:15s}: {total:>12.6f}")
        print(f"  {'TOTAL':15s}: {sum(task_losses.values()):>12.6f}")

    def _check_memory(self):
        """检查GPU显存使用"""
        print("\n[GPU显存]")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            print(f"  GPU {i}: 已分配={allocated:.2f}GB, 已预留={reserved:.2f}GB, "
                  f"峰值={max_allocated:.2f}GB")

    def _check_gradients_summary(self, model):
        """梯度汇总检查"""
        print("\n[梯度汇总]")
        module_grads = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            module_name = name.split('.')[0]
            if module_name not in module_grads:
                module_grads[module_name] = {'has_grad': 0, 'no_grad': 0, 'nan': 0, 'max_norm': 0.0}

            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                module_grads[module_name]['has_grad'] += 1
                module_grads[module_name]['max_norm'] = max(module_grads[module_name]['max_norm'], grad_norm)
                if torch.isnan(param.grad).any():
                    module_grads[module_name]['nan'] += 1
            else:
                module_grads[module_name]['no_grad'] += 1

        for mod, info in sorted(module_grads.items()):
            status = "OK" if info['nan'] == 0 and info['no_grad'] == 0 else "WARN"
            print(f"  {mod:25s}: 有梯度={info['has_grad']:3d}, 无梯度={info['no_grad']:3d}, "
                  f"NaN={info['nan']:2d}, 最大范数={info['max_norm']:.4e}  [{status}]")

    def _save_loss(self, losses: Dict, iteration: int):
        """保存loss到文件"""
        record = {
            'iter': iteration,
            'time': time.time() - self.start_time,
        }
        for k, v in losses.items():
            record[k] = v.item() if isinstance(v, torch.Tensor) else v

        with open(self.loss_log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')


# =============================================================================
# Part 4: 训练速度分析器
# =============================================================================

class TrainingProfiler:
    """分析训练各阶段耗时

    Usage:
        profiler = TrainingProfiler()

        # 在forward_train中插桩:
        profiler.start('track_forward')
        losses_track, outs_track = self.forward_track_train(...)
        profiler.end('track_forward')

        profiler.start('seg_forward')
        losses_seg, outs_seg = self.seg_head.forward_train(...)
        profiler.end('seg_forward')

        profiler.report()
    """

    def __init__(self):
        self.timings = {}
        self.counts = {}
        self._starts = {}

    def start(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._starts[name] = time.time()

    def end(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self._starts[name]
        if name not in self.timings:
            self.timings[name] = 0.0
            self.counts[name] = 0
        self.timings[name] += elapsed
        self.counts[name] += 1

    def report(self):
        print("\n" + "=" * 70)
        print("训练耗时分析")
        print("=" * 70)
        total = sum(self.timings.values())
        for name in sorted(self.timings.keys()):
            avg = self.timings[name] / max(self.counts[name], 1)
            pct = self.timings[name] / max(total, 1e-9) * 100
            print(f"  {name:30s}: 总计={self.timings[name]:8.2f}s, "
                  f"平均={avg*1000:8.1f}ms, 占比={pct:5.1f}%  "
                  f"({self.counts[name]}次)")
        print(f"  {'TOTAL':30s}: {total:.2f}s")
        print("=" * 70)

    def reset(self):
        self.timings.clear()
        self.counts.clear()
        self._starts.clear()


# =============================================================================
# Part 5: 快速验证训练流程 (Mock数据)
# =============================================================================

def create_mock_data_uniad(device='cuda', batch_size=1, queue_length=3, num_cams=6):
    """创建UniAD的mock训练数据用于快速debug

    Usage:
        data = create_mock_data_uniad(device='cuda')
        losses = model.forward_train(**data)
    """
    H, W = 928, 1600  # 图像尺寸
    num_gt = 10        # GT数量
    fut_steps = 12     # 未来轨迹步数
    past_steps = 4     # 历史轨迹步数
    occ_h, occ_w = 200, 200
    planning_steps = 6

    data = dict(
        img=torch.randn(batch_size, queue_length, num_cams, 3, H, W).to(device),
        img_metas=[[{
            'scene_token': 'debug_scene',
            'can_bus': [0.0]*18,
            'lidar2img': [torch.eye(4).numpy() for _ in range(num_cams)],
            'img_shape': [(H, W, 3)] * num_cams,
            'box_type_3d': 'LiDAR',  # placeholder
        } for _ in range(queue_length)] for _ in range(batch_size)],
        gt_bboxes_3d=[[torch.randn(num_gt, 9).to(device) for _ in range(queue_length)] for _ in range(batch_size)],
        gt_labels_3d=[[torch.randint(0, 10, (num_gt,)).to(device) for _ in range(queue_length)] for _ in range(batch_size)],
        gt_inds=[[torch.arange(num_gt).to(device) for _ in range(queue_length)] for _ in range(batch_size)],
        l2g_t=[[torch.zeros(3).to(device) for _ in range(queue_length)] for _ in range(batch_size)],
        l2g_r_mat=[[torch.eye(3).to(device) for _ in range(queue_length)] for _ in range(batch_size)],
        timestamp=[[float(i) * 0.5 for i in range(queue_length)] for _ in range(batch_size)],
        gt_lane_labels=[torch.randint(0, 4, (50,)).to(device) for _ in range(batch_size)],
        gt_lane_bboxes=[torch.randn(50, 4).to(device) for _ in range(batch_size)],
        gt_lane_masks=[torch.randint(0, 2, (50, occ_h, occ_w)).float().to(device) for _ in range(batch_size)],
        gt_fut_traj=[torch.randn(num_gt, fut_steps, 2).to(device) for _ in range(batch_size)],
        gt_fut_traj_mask=[torch.ones(num_gt, fut_steps).to(device) for _ in range(batch_size)],
        gt_past_traj=[torch.randn(num_gt, past_steps, 2).to(device) for _ in range(batch_size)],
        gt_past_traj_mask=[torch.ones(num_gt, past_steps).to(device) for _ in range(batch_size)],
        gt_sdc_bbox=[torch.randn(1, 9).to(device) for _ in range(batch_size)],
        gt_sdc_label=[torch.zeros(1).long().to(device) for _ in range(batch_size)],
        gt_sdc_fut_traj=[torch.randn(1, fut_steps, 2).to(device) for _ in range(batch_size)],
        gt_sdc_fut_traj_mask=[torch.ones(1, fut_steps).to(device) for _ in range(batch_size)],
        gt_segmentation=[torch.randint(0, 2, (5, 1, occ_h, occ_w)).float().to(device) for _ in range(batch_size)],
        gt_instance=[torch.randint(0, 10, (5, 1, occ_h, occ_w)).float().to(device) for _ in range(batch_size)],
        gt_occ_img_is_valid=[torch.ones(5, 1, occ_h, occ_w).bool().to(device) for _ in range(batch_size)],
        sdc_planning=[torch.randn(1, planning_steps, 3).to(device) for _ in range(batch_size)],
        sdc_planning_mask=[torch.ones(1, planning_steps).to(device) for _ in range(batch_size)],
        command=[torch.tensor([1, 0, 0]).float().to(device) for _ in range(batch_size)],
        gt_future_boxes=None,
    )
    return data


def create_mock_data_fusionad(device='cuda', batch_size=1, queue_length=3, num_cams=6):
    """创建FusionAD的mock训练数据 (额外包含LiDAR点云)

    Usage:
        data = create_mock_data_fusionad(device='cuda')
        losses = model.forward_train(**data)
    """
    data = create_mock_data_uniad(device, batch_size, queue_length, num_cams)

    # FusionAD额外需要点云数据
    num_points = 30000
    data['points'] = [[torch.randn(num_points, 5).to(device) for _ in range(queue_length)] for _ in range(batch_size)]

    return data


# =============================================================================
# Part 6: 完整训练Debug流程
# =============================================================================

def debug_training_step(model, data, optimizer=None, device='cuda'):
    """执行一个完整的训练step并输出debug信息

    Usage:
        from mmcv import Config
        from mmdet.models import build_detector

        cfg = Config.fromfile('projects/configs/stage2_e2e/base_e2e.py')
        model = build_detector(cfg.model).to('cuda')

        data = create_mock_data_uniad(device='cuda')
        debug_training_step(model, data)
    """
    model.train()
    profiler = TrainingProfiler()
    debug_hook = TrainingDebugHook(log_interval=1, save_loss_history=False)

    # 1. 参数分析
    print("\n" + "#" * 80)
    print("# Step 1: 模型参数分析")
    print("#" * 80)
    analyze_model_params(model)

    # 2. 前向传播
    print("\n" + "#" * 80)
    print("# Step 2: 前向传播")
    print("#" * 80)

    profiler.start('forward')
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        losses = model.forward_train(**data)
        profiler.end('forward')
        print("[OK] 前向传播成功")
    except Exception as e:
        profiler.end('forward')
        print(f"[FAIL] 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 3. Loss检查
    debug_hook.after_train_iter_manual(model, losses, 0)

    # 4. 反向传播
    print("\n" + "#" * 80)
    print("# Step 3: 反向传播")
    print("#" * 80)

    profiler.start('backward')
    try:
        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor) and v.requires_grad)
        print(f"总Loss: {total_loss.item():.6f}")
        total_loss.backward()
        profiler.end('backward')
        print("[OK] 反向传播成功")
    except Exception as e:
        profiler.end('backward')
        print(f"[FAIL] 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 5. 梯度检查
    print("\n" + "#" * 80)
    print("# Step 4: 梯度流检查")
    print("#" * 80)
    check_gradient_flow(model, total_loss)

    # 6. 优化器步骤
    if optimizer is not None:
        profiler.start('optimizer_step')
        optimizer.step()
        optimizer.zero_grad()
        profiler.end('optimizer_step')
        print("\n[OK] 优化器更新完成")

    # 7. 耗时报告
    profiler.report()

    # 8. 显存报告
    if torch.cuda.is_available():
        print("\n[显存报告]")
        for i in range(torch.cuda.device_count()):
            peak = torch.cuda.max_memory_allocated(i) / 1024**3
            print(f"  GPU {i} 峰值显存: {peak:.2f} GB")

    return losses


# =============================================================================
# Part 7: 加速训练配置生成器
# =============================================================================

def generate_fast_training_config(base_config: str = 'uniad', strategy: str = 'balanced'):
    """生成加速训练的配置建议

    Args:
        base_config: 'uniad' 或 'fusionad'
        strategy: 'balanced'(平衡), 'speed'(最快), 'quality'(保质量)
    """
    configs = {
        'uniad': {
            'balanced': {
                'queue_length': 3,
                'total_epochs': 12,
                'lr': 2e-4,
                'warmup_iters': 300,
                'samples_per_gpu': 1,
                'freeze_img_backbone': True,
                'freeze_img_neck': True,
                'freeze_bn': True,
                'freeze_bev_encoder': True,
                'fp16': True,
                'grad_accumulation': 2,
                'description': 'UniAD平衡模式: 冻结backbone+neck+BEV, FP16, 12 epochs'
            },
            'speed': {
                'queue_length': 1,
                'total_epochs': 6,
                'lr': 5e-4,
                'warmup_iters': 100,
                'samples_per_gpu': 2,
                'freeze_img_backbone': True,
                'freeze_img_neck': True,
                'freeze_bn': True,
                'freeze_bev_encoder': True,
                'fp16': True,
                'grad_accumulation': 1,
                'bev_h': 100,
                'bev_w': 100,
                'description': 'UniAD极速模式: 1帧序列, 小BEV, 6 epochs'
            },
            'quality': {
                'queue_length': 5,
                'total_epochs': 20,
                'lr': 2e-4,
                'warmup_iters': 500,
                'samples_per_gpu': 1,
                'freeze_img_backbone': True,
                'freeze_img_neck': True,
                'freeze_bn': True,
                'freeze_bev_encoder': True,
                'fp16': True,
                'grad_accumulation': 4,
                'description': 'UniAD高质量模式: 5帧序列, 梯度累积4, 20 epochs'
            },
        },
        'fusionad': {
            'balanced': {
                'queue_length': 3,
                'total_epochs': 20,
                'lr': 1e-4,
                'warmup_iters': 1000,
                'freeze_img_modules': True,
                'freeze_bev_encoder': True,
                'freeze_track': True,
                'freeze_seg': True,
                'freeze_motion': True,
                'freeze_occ': False,
                'fp16': True,
                'description': 'FusionAD平衡模式: 冻结track+seg+motion, 训练occ+planning'
            },
            'speed': {
                'queue_length': 1,
                'total_epochs': 10,
                'lr': 3e-4,
                'warmup_iters': 500,
                'freeze_img_modules': True,
                'freeze_bev_encoder': True,
                'freeze_track': True,
                'freeze_seg': True,
                'freeze_motion': True,
                'freeze_occ': True,
                'fp16': True,
                'description': 'FusionAD极速模式: 仅训练planning head'
            },
        },
    }

    config = configs.get(base_config, {}).get(strategy, {})
    if not config:
        print(f"未找到配置: {base_config}/{strategy}")
        return None

    print("\n" + "=" * 70)
    print(f"推荐加速训练配置: {config['description']}")
    print("=" * 70)
    for k, v in config.items():
        if k != 'description':
            print(f"  {k:25s} = {v}")
    print("=" * 70)

    return config


# =============================================================================
# Part 8: 一键Debug入口
# =============================================================================

def run_full_debug(project='uniad', config_path=None, device='cuda'):
    """一键运行完整Debug流程

    Usage (在项目根目录):
        # UniAD
        cd /home/user/UniADmjy
        python /home/user/training_debug_and_analysis.py uniad

        # FusionAD
        cd /home/user/FusionADmjy
        python /home/user/training_debug_and_analysis.py fusionad
    """
    print("=" * 80)
    print(f"训练Debug流程 - {project.upper()}")
    print("=" * 80)

    if project == 'uniad':
        if config_path is None:
            config_path = 'projects/configs/stage2_e2e/base_e2e.py'

        try:
            from mmcv import Config
            from mmdet.models import build_detector

            cfg = Config.fromfile(config_path)
            model = build_detector(cfg.model).to(device)
            data = create_mock_data_uniad(device=device)
            debug_training_step(model, data, device=device)
        except ImportError:
            print("[WARN] mmcv/mmdet未安装, 使用Mock模式")
            print("请确保已安装: pip install mmcv-full mmdet mmdet3d")
            print("\n生成加速训练配置建议:")
            generate_fast_training_config('uniad', 'balanced')
            generate_fast_training_config('uniad', 'speed')

    elif project == 'fusionad':
        if config_path is None:
            config_path = 'projects/configs/stage2_e2e/fusion_base_e2e.py'

        try:
            from mmcv import Config
            from mmdet.models import build_detector

            cfg = Config.fromfile(config_path)
            model = build_detector(cfg.model).to(device)
            data = create_mock_data_fusionad(device=device)
            debug_training_step(model, data, device=device)
        except ImportError:
            print("[WARN] mmcv/mmdet未安装, 使用Mock模式")
            print("\n生成加速训练配置建议:")
            generate_fast_training_config('fusionad', 'balanced')
            generate_fast_training_config('fusionad', 'speed')


if __name__ == '__main__':
    project = sys.argv[1] if len(sys.argv) > 1 else 'uniad'
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    device = sys.argv[3] if len(sys.argv) > 3 else ('cuda' if torch.cuda.is_available() else 'cpu')

    run_full_debug(project, config_path, device)

    # 始终打印训练流程分析
    print("\n\n")
    print(__doc__)
