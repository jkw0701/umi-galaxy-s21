"""
Visualize SLAM trajectories, gripper widths, and Roll/Pitch/Yaw from a dataset.zarr.zip.
3D 궤적, XY/XZ 투영, TCP 이동거리 히스토그램, 그리퍼 폭, Z높이, Roll/Pitch/Yaw 그래프를 출력한다.
visualize_dataset.py의 확장 버전.

Usage:
    python scripts_slam_s21/visualize_dataset_rpy.py -i <dataset.zarr.zip>
    python scripts_slam_s21/visualize_dataset_rpy.py -i <dataset.zarr.zip> -o out.png
    python scripts_slam_s21/visualize_dataset_rpy.py -i <dataset.zarr.zip> --ep 0 5 10
"""

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import pathlib
import click
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imagecodecs  # needed for jpegxl codec
import zarr


def load_zarr(path):
    z = zarr.open(str(path), 'r')
    eef_pos = z['data/robot0_eef_pos'][:]           # (N, 3)
    eef_rot = z['data/robot0_eef_rot_axis_angle'][:] # (N, 3)
    gripper  = z['data/robot0_gripper_width'][:]     # (N, 1)
    ep_ends  = z['meta/episode_ends'][:]             # (n_ep,)
    return eef_pos, eef_rot, gripper, ep_ends


def get_episode_slices(ep_ends):
    starts = np.concatenate([[0], ep_ends[:-1]])
    return [slice(s, e) for s, e in zip(starts, ep_ends)]


def axis_angle_to_rpy(rot_vec):
    """
    rot_vec: (N, 3) axis-angle 벡터 배열
    반환: roll, pitch, yaw 각도 배열 (N,) — 단위: degrees, ZYX Euler 기준
    """
    angles = np.linalg.norm(rot_vec, axis=1, keepdims=True)  # (N, 1)
    safe_angles = np.where(angles < 1e-8, 1.0, angles)
    axes = rot_vec / safe_angles  # (N, 3)

    c = np.cos(angles[:, 0])
    s = np.sin(angles[:, 0])
    t = 1.0 - c
    kx, ky, kz = axes[:, 0], axes[:, 1], axes[:, 2]

    # Rodrigues → 회전행렬 원소
    R00 = t*kx*kx + c
    R01 = t*kx*ky - s*kz
    R02 = t*kx*kz + s*ky
    R10 = t*kx*ky + s*kz
    R20 = t*kx*kz - s*ky
    R21 = t*ky*kz + s*kx
    R22 = t*kz*kz + c

    # ZYX Euler → degrees
    pitch = np.degrees(np.arctan2(-R20, np.sqrt(R00**2 + R10**2)))
    roll  = np.degrees(np.arctan2(R21, R22))
    yaw   = np.degrees(np.arctan2(R10, R00))

    zero_mask = (angles[:, 0] < 1e-8)
    roll[zero_mask] = 0.0
    pitch[zero_mask] = 0.0
    yaw[zero_mask] = 0.0

    return roll, pitch, yaw


def set_axes_equal_3d(ax, all_xyz):
    if not all_xyz:
        return
    xyz = np.vstack(all_xyz)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    if max_range < 1e-6:
        return
    mid = np.array([(x.max()+x.min()), (y.max()+y.min()), (z.max()+z.min())]) / 2.0
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)


@click.command()
@click.option('-i', '--input', 'zarr_path', required=True, help='dataset.zarr.zip path')
@click.option('-o', '--output', default=None, help='Save figure to file')
@click.option('--ep', multiple=True, type=int, default=None,
              help='Episode indices to highlight (default: all)')
@click.option('--equal_scale', is_flag=True, default=False,
              help='추가로 equal-scale 축 그래프를 별도 창으로 표시')
def main(zarr_path, output, ep, equal_scale):
    zarr_path = pathlib.Path(zarr_path)
    eef_pos, eef_rot, gripper, ep_ends = load_zarr(zarr_path)
    slices = get_episode_slices(ep_ends)
    n_ep = len(slices)

    # 전체 데이터에 대해 RPY 미리 계산
    roll_all, pitch_all, yaw_all = axis_angle_to_rpy(eef_rot)

    # 지정 에피소드에 대해서 그래프 출력
    ep_indices = list(ep) if ep else list(range(n_ep))
    print(f"Total episodes: {n_ep},  visualizing: {len(ep_indices)}")

    durations = [slices[i].stop - slices[i].start for i in ep_indices]
    motions   = [np.linalg.norm(eef_pos[slices[i]][-1] - eef_pos[slices[i]][0]) * 1000
                 for i in ep_indices]
    print(f"Frames/ep:  min={min(durations)}  max={max(durations)}  avg={np.mean(durations):.1f}")
    print(f"TCP motion: min={min(motions):.1f}mm  max={max(motions):.1f}mm  avg={np.mean(motions):.1f}mm")

    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(ep_indices), 1))

    # ── 메인 Figure: 3×3 레이아웃 ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))

    ax3d      = fig.add_subplot(331, projection='3d')
    ax_xy     = fig.add_subplot(332)
    ax_xz     = fig.add_subplot(333)
    ax_motion = fig.add_subplot(334)
    ax_gripper= fig.add_subplot(335)
    ax_ztime  = fig.add_subplot(336)
    ax_roll   = fig.add_subplot(337)
    ax_pitch  = fig.add_subplot(338)
    ax_yaw    = fig.add_subplot(339)

    all_pos_for_eq = []

    for ci, i in enumerate(ep_indices):
        s = slices[i]
        pos   = eef_pos[s]
        color = cmap(ci)
        label = f'ep{i}' if len(ep_indices) <= 15 else None
        t_arr = np.arange(len(pos))

        # ── 기존 그래프 (pos: m → mm) ─────────────────────────────────────────
        pos_mm = pos * 1000.0

        ax3d.plot(pos_mm[:,0], pos_mm[:,1], pos_mm[:,2], color=color, lw=0.8, alpha=0.7, label=label)
        ax3d.scatter(*pos_mm[0], color=color, s=20, zorder=5)

        ax_xy.plot(pos_mm[:,0], pos_mm[:,1], color=color, lw=0.8, alpha=0.7, label=label)
        ax_xy.scatter(*pos_mm[0,:2], color=color, s=15, zorder=5)

        ax_xz.plot(pos_mm[:,0], pos_mm[:,2], color=color, lw=0.8, alpha=0.7, label=label)

        gw = gripper[s, 0]
        ax_gripper.plot(t_arr, gw * 1000.0, color=color, lw=0.8, alpha=0.6)

        ax_ztime.plot(t_arr, pos_mm[:,2], color=color, lw=0.8, alpha=0.6)

        # ── RPY 그래프 ───────────────────────────────────────────────────────
        r = roll_all[s]
        p = pitch_all[s]
        y = yaw_all[s]

        ax_roll.plot(t_arr, r, color=color, lw=0.8, alpha=0.7)
        ax_pitch.plot(t_arr, p, color=color, lw=0.8, alpha=0.7)
        ax_yaw.plot(t_arr, y, color=color, lw=0.8, alpha=0.7)

        all_pos_for_eq.append(pos_mm)

    # Motion histogram
    ax_motion.hist(motions, bins=20, color='steelblue', edgecolor='white')
    ax_motion.set_xlabel('TCP motion (mm)')
    ax_motion.set_ylabel('Count')
    ax_motion.set_title(f'TCP motion per episode\n(avg={np.mean(motions):.1f}mm)')
    ax_motion.axvline(np.mean(motions), color='red', linestyle='--', label='mean')
    ax_motion.legend()

    set_axes_equal_3d(ax3d, all_pos_for_eq)
    ax3d.set_xlabel('X (mm)'); ax3d.set_ylabel('Y (mm)'); ax3d.set_zlabel('Z (mm)')
    ax3d.set_title('3D TCP Trajectory')
    if len(ep_indices) <= 15:
        ax3d.legend(fontsize=6, loc='upper left')

    ax_xy.set_xlabel('X (mm)'); ax_xy.set_ylabel('Y (mm)')
    ax_xy.set_title('Top-down (XY)')
    ax_xy.axis('equal'); ax_xy.grid(True, alpha=0.3)

    ax_xz.set_xlabel('X (mm)'); ax_xz.set_ylabel('Z (mm)')
    ax_xz.set_title('Side view (XZ)')
    ax_xz.grid(True, alpha=0.3)

    ax_gripper.set_xlabel('Frame'); ax_gripper.set_ylabel('Width (mm)')
    ax_gripper.set_title('Gripper width over time')
    ax_gripper.grid(True, alpha=0.3)

    ax_ztime.set_xlabel('Frame'); ax_ztime.set_ylabel('Z height (mm)')
    ax_ztime.set_title('Z height over time')
    ax_ztime.grid(True, alpha=0.3)

    ax_roll.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax_roll.set_xlabel('Frame'); ax_roll.set_ylabel('Roll (deg)')
    ax_roll.set_title('Roll over time (ZYX Euler)')
    ax_roll.grid(True, alpha=0.3)

    ax_pitch.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax_pitch.set_xlabel('Frame'); ax_pitch.set_ylabel('Pitch (deg)')
    ax_pitch.set_title('Pitch over time (ZYX Euler)')
    ax_pitch.grid(True, alpha=0.3)

    ax_yaw.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax_yaw.set_xlabel('Frame'); ax_yaw.set_ylabel('Yaw (deg)')
    ax_yaw.set_title('Yaw over time (ZYX Euler)')
    ax_yaw.grid(True, alpha=0.3)

    plt.suptitle(f'{zarr_path.name}  —  {n_ep} episodes  (+ Roll/Pitch/Yaw)', fontsize=12)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved to {output}")

    # ── Figure 2: 6DOF + Gripper 시계열 ─────────────────────────────────────
    fig_dof, axes_dof = plt.subplots(4, 2, figsize=(16, 18))
    ax_x   = axes_dof[0, 0]
    ax_y   = axes_dof[0, 1]
    ax_z   = axes_dof[1, 0]
    ax_r   = axes_dof[1, 1]
    ax_p   = axes_dof[2, 0]
    ax_yw  = axes_dof[2, 1]
    ax_gw  = axes_dof[3, 0]
    axes_dof[3, 1].axis('off')  # 7개라 마지막 칸 비움

    for ci, i in enumerate(ep_indices):
        s = slices[i]
        pos_mm = eef_pos[s] * 1000.0
        gw_mm  = gripper[s, 0] * 1000.0
        r = roll_all[s]
        p = pitch_all[s]
        y = yaw_all[s]
        color = cmap(ci)
        label = f'ep{i}' if len(ep_indices) <= 15 else None
        t_arr = np.arange(len(pos_mm))

        ax_x.plot(t_arr, pos_mm[:, 0], color=color, lw=0.8, alpha=0.7, label=label)
        ax_y.plot(t_arr, pos_mm[:, 1], color=color, lw=0.8, alpha=0.7)
        ax_z.plot(t_arr, pos_mm[:, 2], color=color, lw=0.8, alpha=0.7)
        ax_r.plot(t_arr, r,            color=color, lw=0.8, alpha=0.7)
        ax_p.plot(t_arr, p,            color=color, lw=0.8, alpha=0.7)
        ax_yw.plot(t_arr, y,           color=color, lw=0.8, alpha=0.7)
        ax_gw.plot(t_arr, gw_mm,       color=color, lw=0.8, alpha=0.7)

    for ax, ylabel, title in [
        (ax_x,  'X (mm)',    'X position'),
        (ax_y,  'Y (mm)',    'Y position'),
        (ax_z,  'Z (mm)',    'Z position'),
        (ax_r,  'Roll (deg)',  'Roll'),
        (ax_p,  'Pitch (deg)', 'Pitch'),
        (ax_yw, 'Yaw (deg)',   'Yaw'),
        (ax_gw, 'Width (mm)', 'Gripper width'),
    ]:
        ax.set_xlabel('Frame')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', lw=0.4, linestyle='--')

    if len(ep_indices) <= 15:
        ax_x.legend(fontsize=6, loc='best')

    fig_dof.suptitle(f'{zarr_path.name}  —  6DOF + Gripper time series', fontsize=12)
    fig_dof.tight_layout()

    if output:
        dof_output = pathlib.Path(output).with_stem(pathlib.Path(output).stem + '_6dof')
        fig_dof.savefig(str(dof_output), dpi=150, bbox_inches='tight')
        print(f"Saved to {dof_output}")

    # ── Equal-scale 별도 창 ──────────────────────────────────────────────────
    if equal_scale and all_pos_for_eq:
        fig2 = plt.figure(figsize=(18, 6))
        ax2_3d = fig2.add_subplot(131, projection='3d')
        ax2_xy = fig2.add_subplot(132)
        ax2_xz = fig2.add_subplot(133)

        for ci, i in enumerate(ep_indices):
            s = slices[i]
            pos_mm = eef_pos[s] * 1000.0
            color = cmap(ci)
            label = f'ep{i}' if len(ep_indices) <= 15 else None

            ax2_3d.plot(pos_mm[:,0], pos_mm[:,1], pos_mm[:,2], color=color, lw=0.8, alpha=0.8, label=label)
            ax2_3d.scatter(*pos_mm[0], color=color, s=20, zorder=5)

            ax2_xy.plot(pos_mm[:,0], pos_mm[:,1], color=color, lw=0.8, alpha=0.8, label=label)
            ax2_xy.scatter(pos_mm[0,0], pos_mm[0,1], color=color, s=15, zorder=5)

            ax2_xz.plot(pos_mm[:,0], pos_mm[:,2], color=color, lw=0.8, alpha=0.8)
            ax2_xz.scatter(pos_mm[0,0], pos_mm[0,2], color=color, s=15, zorder=5)

        set_axes_equal_3d(ax2_3d, all_pos_for_eq)

        ax2_3d.set_xlabel('X (mm)'); ax2_3d.set_ylabel('Y (mm)'); ax2_3d.set_zlabel('Z (mm)')
        ax2_3d.set_title('3D TCP  (equal scale)')
        if len(ep_indices) <= 15:
            ax2_3d.legend(fontsize=6, loc='upper left')

        ax2_xy.set_xlabel('X (mm)'); ax2_xy.set_ylabel('Y (mm)')
        ax2_xy.set_title('Top-down XY  (equal scale)')
        ax2_xy.set_aspect('equal'); ax2_xy.grid(True, alpha=0.3)

        ax2_xz.set_xlabel('X (mm)'); ax2_xz.set_ylabel('Z (mm)')
        ax2_xz.set_title('Side view XZ  (equal scale)')
        ax2_xz.set_aspect('equal'); ax2_xz.grid(True, alpha=0.3)

        fig2.suptitle(f'{zarr_path.name}  —  equal scale', fontsize=11)
        fig2.tight_layout()

        if output:
            eq_output = pathlib.Path(output).with_stem(pathlib.Path(output).stem + '_equal_scale')
            fig2.savefig(str(eq_output), dpi=150, bbox_inches='tight')
            print(f"Saved to {eq_output}")

    if not output:
        plt.show()


if __name__ == '__main__':
    main()
