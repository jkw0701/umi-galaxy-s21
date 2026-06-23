"""
Dataset vs Model Prediction Comparison Tool
============================================

두 가지 모드로 실행 가능:

[모드 1] Dataset 시각화만 (체크포인트 없이)
    python tests/eval_dataset_vs_model.py \
        --dataset /home/kist/UMI/s10_video/3_31_test2_pipe/dataset.zarr.zip

[모드 2] Dataset + 모델 예측 비교 (체크포인트 있을 때)
    python tests/eval_dataset_vs_model.py \
        --dataset /home/kist/UMI/s10_video/3_31_test2_pipe/dataset.zarr.zip \
        --checkpoint data/outputs/2026.03.31/13.58.53_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt \
        --episode 0

출력 결과:
  - 3D 위치 궤적 (ground truth vs 예측)
  - X/Y/Z 시간 시계열
  - 회전 (axis-angle) 시계열
  - Gripper 너비 시계열
  - 위치 오차 (MAE, RMSE)
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — plt.show() 없이 파일로만 저장
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import torch
import zarr

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()

from diffusion_policy.common.replay_buffer import ReplayBuffer


# ──────────────────────────────────────────────────────
# Rotation utilities
# ──────────────────────────────────────────────────────

def rotation_6d_to_axis_angle(rot_6d: np.ndarray) -> np.ndarray:
    """rotation_6d (N,6) → axis_angle (N,3)"""
    from scipy.spatial.transform import Rotation
    out = np.zeros((len(rot_6d), 3))
    for i, r in enumerate(rot_6d):
        a = r[:3]
        b = r[3:]
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b - np.dot(b, a) * a
        b = b / (np.linalg.norm(b) + 1e-8)
        c = np.cross(a, b)
        R = np.stack([a, b, c], axis=1)
        out[i] = Rotation.from_matrix(R).as_rotvec()
    return out


def axis_angle_to_rotation_6d(axis_angle: np.ndarray) -> np.ndarray:
    """axis_angle (N,3) → rotation_6d (N,6)"""
    from scipy.spatial.transform import Rotation
    out = np.zeros((len(axis_angle), 6))
    for i, aa in enumerate(axis_angle):
        R = Rotation.from_rotvec(aa).as_matrix()
        out[i] = np.concatenate([R[:, 0], R[:, 1]])
    return out


# ──────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────

def load_dataset(zarr_path: str):
    """Load zarr dataset and return ReplayBuffer."""
    print(f"Loading dataset: {zarr_path}")
    try:
        buff = ReplayBuffer.create_from_path(zarr_path, mode='r')
        print(f"  Episodes: {buff.n_episodes}, Total frames: {buff.n_steps}")
        return buff
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def get_episode_data(buff: ReplayBuffer, episode_idx: int) -> dict:
    """Extract one episode's data."""
    if episode_idx >= buff.n_episodes:
        raise ValueError(f"Episode {episode_idx} not found. Max: {buff.n_episodes-1}")

    ep = buff.get_episode(episode_idx)
    data = {}

    # position: (T, 3)
    data['eef_pos'] = np.array(ep['robot0_eef_pos'])

    # rotation axis-angle: (T, 3)
    data['eef_rot'] = np.array(ep['robot0_eef_rot_axis_angle'])

    # gripper width: (T,) or (T,1)
    gw = np.array(ep['robot0_gripper_width'])
    data['gripper'] = gw.squeeze(-1) if gw.ndim == 2 else gw

    # demo start/end pose
    if 'robot0_demo_start_pose' in ep:
        data['demo_start'] = np.array(ep['robot0_demo_start_pose'])
    if 'robot0_demo_end_pose' in ep:
        data['demo_end'] = np.array(ep['robot0_demo_end_pose'])

    T = len(data['eef_pos'])
    data['timestep'] = np.arange(T)
    print(f"  Episode {episode_idx}: {T} frames")
    return data


# ──────────────────────────────────────────────────────
# Model loading & inference
# ──────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: str):
    """Load trained DiffusionPolicy model from checkpoint."""
    import dill
    import hydra
    from omegaconf import OmegaConf

    print(f"\nLoading model from: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location='cpu', pickle_module=dill)

    cfg = payload['cfg']
    cls_name = cfg['_target_']

    # dynamically import workspace class
    module_path, class_name = cls_name.rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    WorkspaceCls = getattr(module, class_name)

    workspace = WorkspaceCls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema and hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("  Using EMA model")

    policy.num_inference_steps = 16  # DDIM inference iterations (eval_real.py 와 동일)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.to(device)
    policy.eval()
    print(f"  Model loaded on {device}")
    return policy, cfg, device


def run_inference_on_episode(policy, cfg, device, buff: ReplayBuffer,
                              episode_idx: int, obs_horizon: int = 2,
                              action_horizon: int = 16):
    """
    Run model inference on every frame of an episode.

    eval_real.py 와 동일한 전처리 사용:
      - get_real_umi_obs_dict() 로 obs_pose_repr(relative) 변환 후 모델에 입력
      - get_real_umi_action()   로 action_pose_repr(relative) → 절대 좌표 변환

    이렇게 해야 모델 출력(상대 좌표)을 데이터셋(절대 좌표)과 올바르게 비교 가능.
    """
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()
    import torch.nn.functional as F
    from diffusion_policy.common.pytorch_util import dict_apply
    from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action

    obs_pose_repr    = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    shape_meta       = cfg.task.shape_meta

    ep = buff.get_episode(episode_idx)
    T = len(ep['robot0_eef_pos'])

    pos_gt = np.array(ep['robot0_eef_pos'])               # (T, 3)  절대좌표
    rot_gt = np.array(ep['robot0_eef_rot_axis_angle'])    # (T, 3)  axis-angle
    gw_gt  = np.array(ep['robot0_gripper_width']).squeeze(-1)  # (T,)

    # 에피소드 시작 포즈 (get_real_umi_obs_dict의 episode_start_pose 용)
    start_pose = np.concatenate([pos_gt[0], rot_gt[0]])   # (6,)
    episode_start_pose = [start_pose]

    has_image = 'camera0_rgb' in ep
    all_imgs  = np.array(ep['camera0_rgb']) if has_image else None  # (T, H, W, 3) uint8

    pred_pos_list = []
    pred_rot_list = []
    pred_gw_list  = []
    valid_mask    = []

    print(f"\nRunning inference on episode {episode_idx} ({T} frames)  "
          f"[obs={obs_pose_repr}, action={action_pose_repr}]...")

    with torch.no_grad():
        for t in range(T):
            t_start = max(0, t - obs_horizon + 1)

            # ── env_obs 구성 (T, ...) float32, eval_real.py 와 동일 형태 ──
            pos_obs = pos_gt[t_start:t+1]
            rot_obs = rot_gt[t_start:t+1]
            gw_obs  = gw_gt[t_start:t+1]

            pad = obs_horizon - len(pos_obs)
            if pad > 0:
                pos_obs = np.concatenate([np.tile(pos_obs[:1], (pad,1)), pos_obs])
                rot_obs = np.concatenate([np.tile(rot_obs[:1], (pad,1)), rot_obs])
                gw_obs  = np.concatenate([np.tile(gw_obs[:1],  (pad,)),  gw_obs])

            # 이미지
            if has_image:
                imgs = all_imgs[t_start:t+1]
                if pad > 0:
                    imgs = np.concatenate([np.tile(imgs[:1], (pad,1,1,1)), imgs])
                imgs = imgs.astype(np.float32) / 255.0   # (H, h, w, 3)
            else:
                h, w = 224, 224
                imgs = np.zeros((obs_horizon, h, w, 3), dtype=np.float32)

            env_obs = {
                'robot0_eef_pos':              pos_obs,          # (H, 3)
                'robot0_eef_rot_axis_angle':   rot_obs,          # (H, 3)
                'robot0_gripper_width':        gw_obs[:, None],  # (H, 1)
                'camera0_rgb':                 imgs,             # (H, h, w, 3)
            }

            # ── eval_real.py 와 동일한 obs 전처리 ──
            obs_dict_np = get_real_umi_obs_dict(
                env_obs=env_obs,
                shape_meta=shape_meta,
                obs_pose_repr=obs_pose_repr,
                tx_robot1_robot0=None,
                episode_start_pose=episode_start_pose,
            )

            # camera resize (224×224)
            cam = obs_dict_np['camera0_rgb']   # (H, 3, h, w)
            _, _, h, w = cam.shape
            if h != 224 or w != 224:
                cam_t = torch.from_numpy(cam)
                cam_t = F.interpolate(cam_t, size=(224,224), mode='bilinear', align_corners=False)
                obs_dict_np['camera0_rgb'] = cam_t.numpy()

            obs_dict = dict_apply(obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

            try:
                result = policy.predict_action(obs_dict)
                # action_pred: (1, action_horizon, 10)  — 상대좌표 rotation_6d
                raw_action = result['action_pred'][0].cpu().numpy()  # (action_horizon, 10)

                # ── get_real_umi_action: 상대 → 절대 좌표 변환 ──
                action_abs = get_real_umi_action(raw_action, env_obs, action_pose_repr)
                # action_abs: (action_horizon, 7) = pos(3) + rotvec(3) + grip(1)

                pred_pos_list.append(action_abs[0, :3])
                pred_rot_list.append(action_abs[0, 3:6])
                pred_gw_list.append(action_abs[0, 6])
                valid_mask.append(True)
            except Exception as e:
                pred_pos_list.append(pos_gt[t])
                pred_rot_list.append(rot_gt[t])
                pred_gw_list.append(gw_gt[t])
                valid_mask.append(False)
                if t == 0:
                    print(f"  Warning: inference error at t=0: {e}")

    pred_pos = np.array(pred_pos_list)  # (T, 3)
    pred_rot = np.array(pred_rot_list)  # (T, 3)
    pred_gw  = np.array(pred_gw_list)   # (T,)

    return {
        'gt_pos': pos_gt, 'gt_rot': rot_gt, 'gt_gripper': gw_gt,
        'pred_pos': pred_pos, 'pred_rot': pred_rot, 'pred_gripper': pred_gw,
        'valid': np.array(valid_mask),
    }


# ──────────────────────────────────────────────────────
# Plotting: Dataset only
# ──────────────────────────────────────────────────────

def plot_dataset_only(data: dict, episode_idx: int, save_path: str = None):
    """Visualize dataset ground truth trajectories."""
    pos = data['eef_pos']
    rot = data['eef_rot']
    gw  = data['gripper']
    T   = len(pos)
    t   = np.arange(T)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Human Demonstration Trajectory — Episode {episode_idx}  ({T} frames)\n'
                 f'(DROID-SLAM tracked trajectory from human demo)', fontsize=14)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 3D trajectory ──
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=t, cmap='viridis', s=8, alpha=0.8)
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', alpha=0.3, linewidth=0.8)
    ax3d.scatter(*pos[0], color='green', s=120, marker='o', label='Start', zorder=5)
    ax3d.scatter(*pos[-1], color='red',  s=120, marker='X', label='End',   zorder=5)
    plt.colorbar(sc, ax=ax3d, label='Frame', pad=0.1, shrink=0.6)
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory (EEF Position)')
    ax3d.legend(fontsize=8)
    _set_axes_equal_3d(ax3d, pos)

    # ── Position time series ──
    labels_pos = ['X', 'Y', 'Z']
    colors_pos = ['#e74c3c', '#2ecc71', '#3498db']
    for i, (lbl, col) in enumerate(zip(labels_pos, colors_pos)):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, pos[:, i], color=col, linewidth=1.5)
        ax.set_ylabel(f'{lbl} (m)', fontsize=9)
        ax.set_title(f'EEF Position — {lbl}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel('Frame')

    # ── Rotation + Gripper ──
    ax_rot = fig.add_subplot(gs[0, 2])
    rot_mag = np.linalg.norm(rot, axis=1)
    ax_rot.plot(t, rot[:, 0], label='rx', linewidth=1.2)
    ax_rot.plot(t, rot[:, 1], label='ry', linewidth=1.2)
    ax_rot.plot(t, rot[:, 2], label='rz', linewidth=1.2)
    ax_rot.plot(t, rot_mag, 'k--', label='|r|', linewidth=1.0, alpha=0.6)
    ax_rot.set_ylabel('rad', fontsize=9)
    ax_rot.set_title('Rotation (Axis-Angle)', fontsize=9)
    ax_rot.legend(fontsize=7); ax_rot.grid(True, alpha=0.3)

    ax_gw = fig.add_subplot(gs[1, 2])
    ax_gw.plot(t, gw, color='#9b59b6', linewidth=1.5)
    ax_gw.set_ylabel('Width (m)', fontsize=9)
    ax_gw.set_title('Gripper Width', fontsize=9)
    ax_gw.grid(True, alpha=0.3)

    # ── Statistics table ──
    ax_stat = fig.add_subplot(gs[2, 2])
    ax_stat.axis('off')
    stats = [
        ['Metric', 'X', 'Y', 'Z'],
        ['Mean (m)', f'{pos[:,0].mean():.4f}', f'{pos[:,1].mean():.4f}', f'{pos[:,2].mean():.4f}'],
        ['Std  (m)', f'{pos[:,0].std():.4f}',  f'{pos[:,1].std():.4f}',  f'{pos[:,2].std():.4f}'],
        ['Min  (m)', f'{pos[:,0].min():.4f}',  f'{pos[:,1].min():.4f}',  f'{pos[:,2].min():.4f}'],
        ['Max  (m)', f'{pos[:,0].max():.4f}',  f'{pos[:,1].max():.4f}',  f'{pos[:,2].max():.4f}'],
        ['Range(m)', f'{pos[:,0].ptp():.4f}',  f'{pos[:,1].ptp():.4f}',  f'{pos[:,2].ptp():.4f}'],
        ['Frames',  str(T), '', ''],
        ['Gripper', f'min={gw.min():.4f}', f'max={gw.max():.4f}', ''],
    ]
    tbl = ax_stat.table(cellText=stats[1:], colLabels=stats[0],
                        loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax_stat.set_title('Dataset Statistics', fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close('all')


# ──────────────────────────────────────────────────────
# Plotting: Comparison (GT vs Predicted)
# ──────────────────────────────────────────────────────

def plot_comparison(result: dict, episode_idx: int, save_path: str = None):
    """Compare model predictions vs ground truth."""
    gt_pos  = result['gt_pos']
    gt_rot  = result['gt_rot']
    gt_gw   = result['gt_gripper']
    pred_pos = result['pred_pos']
    pred_rot = result['pred_rot']
    pred_gw  = result['pred_gripper']
    T = len(gt_pos)
    t = np.arange(T)

    # Error computation
    pos_err = np.linalg.norm(gt_pos - pred_pos, axis=1)  # (T,)
    pos_mae  = float(np.mean(pos_err))
    pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))

    per_axis_mae = np.abs(gt_pos - pred_pos).mean(axis=0)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'Policy Prediction vs Human Demonstration — Episode {episode_idx}  ({T} frames)\n'
                 f'Human Demo: DROID-SLAM tracked trajectory  |  Policy Pred: Diffusion Policy output\n'
                 f'Position MAE={pos_mae*100:.2f} cm  RMSE={pos_rmse*100:.2f} cm', fontsize=12)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── 3D trajectory comparison ──
    ax3d = fig.add_subplot(gs[:2, 0], projection='3d')
    ax3d.plot(gt_pos[:,0], gt_pos[:,1], gt_pos[:,2],
              'b-', linewidth=1.5, alpha=0.8, label='Human Demo')
    ax3d.plot(pred_pos[:,0], pred_pos[:,1], pred_pos[:,2],
              'r--', linewidth=1.5, alpha=0.8, label='Policy Pred')
    ax3d.scatter(*gt_pos[0],   color='green', s=100, marker='o', zorder=5)
    ax3d.scatter(*gt_pos[-1],  color='blue',  s=100, marker='X', zorder=5)
    ax3d.scatter(*pred_pos[0], color='orange',s=100, marker='o', zorder=5)
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title('3D Trajectory')
    ax3d.legend(fontsize=8)
    _set_axes_equal_3d(ax3d, np.concatenate([gt_pos, pred_pos]))

    # ── Position time series (X, Y, Z) ──
    labels = ['X', 'Y', 'Z']
    for i, lbl in enumerate(labels):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, gt_pos[:,i],   'b-',  linewidth=1.5, label='Human Demo',  alpha=0.85)
        ax.plot(t, pred_pos[:,i], 'r--', linewidth=1.5, label='Policy Pred', alpha=0.85)
        ax.set_ylabel(f'{lbl} (m)', fontsize=9)
        ax.set_title(f'Position — {lbl}  (MAE={per_axis_mae[i]*100:.2f} cm)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel('Frame')

    # ── Position error over time ──
    ax_err = fig.add_subplot(gs[3, 1])
    ax_err.fill_between(t, 0, pos_err * 100, alpha=0.4, color='red')
    ax_err.plot(t, pos_err * 100, 'r-', linewidth=1.2)
    ax_err.axhline(pos_mae * 100, color='darkred', linestyle='--',
                   linewidth=1.0, label=f'MAE={pos_mae*100:.2f} cm')
    ax_err.set_ylabel('Error (cm)', fontsize=9)
    ax_err.set_xlabel('Frame')
    ax_err.set_title('Position Error (L2) over Time', fontsize=9)
    ax_err.legend(fontsize=7)
    ax_err.grid(True, alpha=0.3)

    # ── Rotation comparison ──
    rot_labels = ['rx', 'ry', 'rz']
    for i, rl in enumerate(rot_labels):
        ax = fig.add_subplot(gs[i, 2])
        ax.plot(t, gt_rot[:,i],   'b-',  linewidth=1.2, label='Human Demo',  alpha=0.85)
        ax.plot(t, pred_rot[:,i], 'r--', linewidth=1.2, label='Policy Pred', alpha=0.85)
        ax.set_ylabel('rad', fontsize=9)
        ax.set_title(f'Rotation — {rl}', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel('Frame')

    # ── Gripper comparison ──
    ax_gw = fig.add_subplot(gs[3, 2])
    ax_gw.plot(t, gt_gw,   'b-',  linewidth=1.5, label='Human Demo',  alpha=0.85)
    ax_gw.plot(t, pred_gw, 'r--', linewidth=1.5, label='Policy Pred', alpha=0.85)
    gw_mae = float(np.abs(gt_gw - pred_gw).mean())
    ax_gw.set_ylabel('Width (m)', fontsize=9)
    ax_gw.set_title(f'Gripper Width  (MAE={gw_mae*1000:.1f} mm)', fontsize=9)
    ax_gw.legend(fontsize=7)
    ax_gw.set_xlabel('Frame')
    ax_gw.grid(True, alpha=0.3)

    # ── Summary stats (bottom-left) ──
    ax_stat = fig.add_subplot(gs[3, 0])
    ax_stat.axis('off')
    stats = [
        ['Metric',        'Value'],
        ['Frames',        str(T)],
        ['Pos MAE',       f'{pos_mae*100:.3f} cm'],
        ['Pos RMSE',      f'{pos_rmse*100:.3f} cm'],
        ['MAE X',         f'{per_axis_mae[0]*100:.3f} cm'],
        ['MAE Y',         f'{per_axis_mae[1]*100:.3f} cm'],
        ['MAE Z',         f'{per_axis_mae[2]*100:.3f} cm'],
        ['Gripper MAE',   f'{gw_mae*1000:.2f} mm'],
        ['Max Pos Err',   f'{pos_err.max()*100:.3f} cm'],
    ]
    tbl = ax_stat.table(cellText=stats[1:], colLabels=stats[0],
                        loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    ax_stat.set_title('Error Summary', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    # Print summary to console
    print("\n" + "="*50)
    print(f"  COMPARISON SUMMARY — Episode {episode_idx}")
    print("="*50)
    print(f"  Frames:       {T}")
    print(f"  Pos MAE:      {pos_mae*100:.3f} cm")
    print(f"  Pos RMSE:     {pos_rmse*100:.3f} cm")
    print(f"  MAE (X/Y/Z):  {per_axis_mae[0]*100:.3f} / {per_axis_mae[1]*100:.3f} / {per_axis_mae[2]*100:.3f} cm")
    print(f"  Gripper MAE:  {gw_mae*1000:.2f} mm")
    print(f"  Max Pos Err:  {pos_err.max()*100:.3f} cm")
    print("="*50)

    plt.close('all')


# ──────────────────────────────────────────────────────
# Summary plot across all episodes
# ──────────────────────────────────────────────────────

def _plot_all_episodes_summary(all_results: dict, save_path: str):
    """
    전체 에피소드의 오차를 한 장으로 요약합니다.

    - 에피소드별 Pos MAE / RMSE / Gripper MAE 막대 그래프
    - 에피소드별 X/Y/Z MAE 누적 막대
    - 전체 통계 테이블
    """
    ep_ids   = sorted(all_results.keys())
    mae_list = []
    rmse_list = []
    xyz_mae_list = []
    gw_mae_list = []

    for ep_idx in ep_ids:
        r = all_results[ep_idx]
        pos_err = np.linalg.norm(r['gt_pos'] - r['pred_pos'], axis=1)
        mae_list.append(pos_err.mean() * 100)
        rmse_list.append(np.sqrt((pos_err**2).mean()) * 100)
        xyz_mae_list.append(np.abs(r['gt_pos'] - r['pred_pos']).mean(axis=0) * 100)
        gw_mae_list.append(np.abs(r['gt_gripper'] - r['pred_gripper']).mean() * 1000)

    mae_arr  = np.array(mae_list)
    rmse_arr = np.array(rmse_list)
    xyz_arr  = np.array(xyz_mae_list)   # (N, 3)
    gw_arr   = np.array(gw_mae_list)
    x        = np.arange(len(ep_ids))

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(ep_ids) * 0.4), 14))
    fig.suptitle(
        f'All Episodes Summary  ({len(ep_ids)} episodes)\n'
        f'Mean Pos MAE={mae_arr.mean():.2f} cm  ±{mae_arr.std():.2f}  |  '
        f'Gripper MAE={gw_arr.mean():.1f} mm',
        fontsize=13)

    # ── 1. Pos MAE & RMSE ──
    ax = axes[0]
    ax.bar(x - 0.2, mae_arr,  0.38, label='MAE (cm)',  color='#3498db', alpha=0.85)
    ax.bar(x + 0.2, rmse_arr, 0.38, label='RMSE (cm)', color='#e74c3c', alpha=0.85)
    ax.axhline(mae_arr.mean(),  color='#2980b9', linestyle='--', linewidth=1.0,
               label=f'Mean MAE={mae_arr.mean():.2f} cm')
    ax.axhline(rmse_arr.mean(), color='#c0392b', linestyle='--', linewidth=1.0,
               label=f'Mean RMSE={rmse_arr.mean():.2f} cm')
    ax.set_xticks(x); ax.set_xticklabels([f'ep{i}' for i in ep_ids], fontsize=7, rotation=45)
    ax.set_ylabel('Error (cm)'); ax.set_title('Position Error per Episode'); ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    # ── 2. X/Y/Z MAE stacked ──
    ax2 = axes[1]
    ax2.bar(x,          xyz_arr[:, 0], 0.6, label='MAE X', color='#e74c3c', alpha=0.8)
    ax2.bar(x, xyz_arr[:, 1], 0.6, bottom=xyz_arr[:, 0], label='MAE Y', color='#2ecc71', alpha=0.8)
    ax2.bar(x, xyz_arr[:, 2], 0.6, bottom=xyz_arr[:, 0]+xyz_arr[:, 1], label='MAE Z', color='#3498db', alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels([f'ep{i}' for i in ep_ids], fontsize=7, rotation=45)
    ax2.set_ylabel('MAE (cm)'); ax2.set_title('Per-Axis Position MAE (X/Y/Z stacked)')
    ax2.legend(fontsize=8); ax2.grid(True, axis='y', alpha=0.3)

    # ── 3. Gripper MAE ──
    ax3 = axes[2]
    ax3.bar(x, gw_arr, 0.6, color='#9b59b6', alpha=0.85)
    ax3.axhline(gw_arr.mean(), color='#6c3483', linestyle='--', linewidth=1.0,
                label=f'Mean={gw_arr.mean():.1f} mm')
    ax3.set_xticks(x); ax3.set_xticklabels([f'ep{i}' for i in ep_ids], fontsize=7, rotation=45)
    ax3.set_ylabel('MAE (mm)'); ax3.set_title('Gripper Width MAE per Episode')
    ax3.legend(fontsize=8); ax3.grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')

    # 콘솔 요약
    print("\n" + "="*60)
    print(f"  FULL SESSION SUMMARY  ({len(ep_ids)} episodes)")
    print("="*60)
    print(f"  Pos MAE  : {mae_arr.mean():.3f} ± {mae_arr.std():.3f} cm  "
          f"(min={mae_arr.min():.3f}  max={mae_arr.max():.3f})")
    print(f"  Pos RMSE : {rmse_arr.mean():.3f} ± {rmse_arr.std():.3f} cm")
    print(f"  MAE X    : {xyz_arr[:,0].mean():.3f} cm")
    print(f"  MAE Y    : {xyz_arr[:,1].mean():.3f} cm")
    print(f"  MAE Z    : {xyz_arr[:,2].mean():.3f} cm")
    print(f"  Gripper  : {gw_arr.mean():.2f} ± {gw_arr.std():.2f} mm")
    # worst episodes
    worst5 = np.argsort(mae_arr)[-5:][::-1]
    print(f"  Worst 5 episodes: {[ep_ids[i] for i in worst5]}")
    print("="*60)


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def _set_axes_equal_3d(ax, pos: np.ndarray):
    """Set equal aspect ratio for 3D axes."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    ranges = np.array([x.ptp(), y.ptp(), z.ptp()])
    max_range = max(ranges.max() / 2.0, 0.01)
    mid = np.array([x.mean(), y.mean(), z.mean()])
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Dataset vs Model Prediction Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dataset 시각화만 (체크포인트 없이):
  python tests/eval_dataset_vs_model.py \\
      --dataset /home/kist/UMI/s10_video/3_31_test2_pipe/dataset.zarr.zip

  # 학습된 모델과 비교:
  python tests/eval_dataset_vs_model.py \\
      --dataset /home/kist/UMI/s10_video/3_31_test2_pipe/dataset.zarr.zip \\
      --checkpoint data/outputs/.../checkpoints/latest.ckpt
        """)

    parser.add_argument('--dataset', '-d', required=True,
                        help='Path to dataset.zarr.zip')
    parser.add_argument('--checkpoint', '-c', default=None,
                        help='Path to .ckpt file (optional). '
                             'If not given, only dataset is visualized.')
    parser.add_argument('--episode', '-e', type=int, default=0,
                        help='Episode index to visualize (default: 0)')
    parser.add_argument('--all_episodes', '-a', action='store_true',
                        help='Visualize all episodes (dataset mode only)')
    parser.add_argument('--save', '-s', default=None,
                        help='Save plot to this path (e.g. plot.png)')
    parser.add_argument('--obs_horizon', type=int, default=2,
                        help='Observation horizon used during training (default: 2)')
    parser.add_argument('--action_horizon', type=int, default=16,
                        help='Action horizon used during training (default: 16)')
    args = parser.parse_args()

    # ── 자동 저장 경로: --save 없으면 dataset 옆 폴더에 저장 ──
    dataset_dir = os.path.dirname(os.path.abspath(args.dataset))
    out_dir = os.path.join(dataset_dir, 'eval_plots')
    os.makedirs(out_dir, exist_ok=True)

    # ── Load dataset ──
    buff = load_dataset(args.dataset)

    print(f"\n  Dataset info:")
    print(f"    Episodes: {buff.n_episodes}")
    print(f"    Total frames: {buff.n_steps}")
    for i in range(buff.n_episodes):
        ep = buff.get_episode(i)
        n = len(ep['robot0_eef_pos'])
        print(f"    Episode {i}: {n} frames")

    if args.checkpoint is None:
        # ── Dataset visualization only ──
        print("\n[Mode: Dataset Visualization Only]")
        print("  체크포인트가 없으므로 ground truth만 시각화합니다.")
        print("  학습 완료 후 --checkpoint 옵션을 추가하면 모델 예측과 비교할 수 있습니다.\n")

        if args.all_episodes:
            episodes = list(range(buff.n_episodes))
        else:
            episodes = [args.episode]

        for ep_idx in episodes:
            data = get_episode_data(buff, ep_idx)
            if args.save:
                base, ext = os.path.splitext(args.save)
                save_path = f"{base}_ep{ep_idx}{ext}" if len(episodes) > 1 else args.save
            else:
                save_path = os.path.join(out_dir, f'dataset_ep{ep_idx:03d}.png')
            plot_dataset_only(data, ep_idx, save_path=save_path)
            print(f"  → {save_path}")

        print(f"\n모든 플롯 저장 완료: {out_dir}/")

    else:
        # ── Comparison mode ──
        print("\n[Mode: Dataset vs Model Prediction Comparison]")
        if not os.path.isfile(args.checkpoint):
            print(f"Error: checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        policy, cfg, device = load_model_from_checkpoint(args.checkpoint)

        episodes = list(range(buff.n_episodes)) if args.all_episodes else [args.episode]
        all_results = {}

        for i, ep_idx in enumerate(episodes):
            print(f"\n[{i+1}/{len(episodes)}] Episode {ep_idx}")
            result = run_inference_on_episode(
                policy, cfg, device, buff, ep_idx,
                obs_horizon=args.obs_horizon,
                action_horizon=args.action_horizon
            )
            all_results[ep_idx] = result

            if args.save:
                base, ext = os.path.splitext(args.save)
                save_path = f"{base}_ep{ep_idx}{ext}" if len(episodes) > 1 else args.save
            else:
                save_path = os.path.join(out_dir, f'compare_ep{ep_idx:03d}.png')
            plot_comparison(result, ep_idx, save_path=save_path)
            print(f"  → {save_path}")

        # ── 전체 에피소드 요약 플롯 (all_episodes 시) ──
        if len(episodes) > 1:
            summary_path = os.path.join(out_dir, 'summary_all_episodes.png')
            _plot_all_episodes_summary(all_results, summary_path)
            print(f"\n요약 플롯 → {summary_path}")

        print(f"\n모든 플롯 저장 완료: {out_dir}/")


if __name__ == '__main__':
    main()
