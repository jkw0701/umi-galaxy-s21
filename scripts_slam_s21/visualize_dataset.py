"""
Visualize SLAM trajectories and gripper widths from a dataset.zarr.zip.
eef_pos 궤적, 그리퍼 폭, TCP 이동거리 히스토그램을 출력한다.

Usage:
    python scripts_slam_s21/visualize_dataset.py -i <dataset.zarr.zip>
    python scripts_slam_s21/visualize_dataset.py -i <dataset.zarr.zip> -o out.png
    python scripts_slam_s21/visualize_dataset.py -i <dataset.zarr.zip> --ep 0 5 10
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
    eef_pos = z['data/robot0_eef_pos'][:]          # (N, 3)
    eef_rot = z['data/robot0_eef_rot_axis_angle'][:] # (N, 3)
    gripper  = z['data/robot0_gripper_width'][:]    # (N, 1)
    ep_ends  = z['meta/episode_ends'][:]            # (n_ep,)
    return eef_pos, eef_rot, gripper, ep_ends


def get_episode_slices(ep_ends):
    starts = np.concatenate([[0], ep_ends[:-1]])
    return [slice(s, e) for s, e in zip(starts, ep_ends)]


def set_axes_equal_3d(ax, all_xyz):
    """3D 축 스케일을 실제 비율에 맞게 동일하게 설정."""
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

    ep_indices = list(ep) if ep else list(range(n_ep))
    print(f"Total episodes: {n_ep},  visualizing: {len(ep_indices)}")

    # Per-episode stats
    durations = [slices[i].stop - slices[i].start for i in range(n_ep)]
    motions   = [np.linalg.norm(eef_pos[slices[i]][-1] - eef_pos[slices[i]][0]) * 100
                 for i in range(n_ep)]
    print(f"Frames/ep:  min={min(durations)}  max={max(durations)}  avg={np.mean(durations):.1f}")
    print(f"TCP motion: min={min(motions):.1f}cm  max={max(motions):.1f}cm  avg={np.mean(motions):.1f}cm")

    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(ep_indices), 1))
    fig = plt.figure(figsize=(16, 10))

    ax3d = fig.add_subplot(231, projection='3d')
    ax_xy = fig.add_subplot(232)
    ax_xz = fig.add_subplot(233)
    ax_motion = fig.add_subplot(234)
    ax_gripper = fig.add_subplot(235)
    ax_ztime   = fig.add_subplot(236)

    all_pos_for_eq = []

    for ci, i in enumerate(ep_indices):
        s = slices[i]
        pos = eef_pos[s]
        color = cmap(ci)
        label = f'ep{i}' if len(ep_indices) <= 15 else None

        ax3d.plot(pos[:,0], pos[:,1], pos[:,2], color=color, lw=0.8, alpha=0.7, label=label)
        ax3d.scatter(*pos[0], color=color, s=20, zorder=5)

        ax_xy.plot(pos[:,0], pos[:,1], color=color, lw=0.8, alpha=0.7, label=label)
        ax_xy.scatter(*pos[0,:2], color=color, s=15, zorder=5)

        ax_xz.plot(pos[:,0], pos[:,2], color=color, lw=0.8, alpha=0.7, label=label)

        gw = gripper[s, 0]
        t = np.arange(len(gw))
        ax_gripper.plot(t, gw * 100, color=color, lw=0.8, alpha=0.6)

        ax_ztime.plot(np.arange(len(pos)), pos[:,2] * 100, color=color, lw=0.8, alpha=0.6)

        all_pos_for_eq.append(pos)

    # Motion histogram
    ax_motion.hist(motions, bins=20, color='steelblue', edgecolor='white')
    ax_motion.set_xlabel('TCP motion (cm)')
    ax_motion.set_ylabel('Count')
    ax_motion.set_title(f'TCP motion per episode\n(avg={np.mean(motions):.1f}cm)')
    ax_motion.axvline(np.mean(motions), color='red', linestyle='--', label=f'mean')
    ax_motion.legend()

    set_axes_equal_3d(ax3d, all_pos_for_eq)
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D TCP Trajectory (equal scale)')
    if len(ep_indices) <= 15:
        ax3d.legend(fontsize=6, loc='upper left')

    ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top-down (XY)')
    ax_xy.axis('equal'); ax_xy.grid(True, alpha=0.3)

    ax_xz.set_xlabel('X (m)'); ax_xz.set_ylabel('Z (m)')
    ax_xz.set_title('Side view (XZ)')
    ax_xz.grid(True, alpha=0.3)

    ax_gripper.set_xlabel('Frame'); ax_gripper.set_ylabel('Width (cm)')
    ax_gripper.set_title('Gripper width over time')
    ax_gripper.grid(True, alpha=0.3)

    ax_ztime.set_xlabel('Frame'); ax_ztime.set_ylabel('Z height (cm)')
    ax_ztime.set_title('Z height over time (arm up/down)')
    ax_ztime.grid(True, alpha=0.3)

    plt.suptitle(f'{zarr_path.name}  —  {n_ep} episodes', fontsize=12)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved to {output}")

    # ── Equal-scale 별도 창 ──────────────────────────────────────────────────
    if equal_scale and all_pos_for_eq:
        fig2 = plt.figure(figsize=(18, 6))
        ax2_3d = fig2.add_subplot(131, projection='3d')
        ax2_xy = fig2.add_subplot(132)
        ax2_xz = fig2.add_subplot(133)

        for ci, i in enumerate(ep_indices):
            s = slices[i]
            pos = eef_pos[s]
            color = cmap(ci)
            label = f'ep{i}' if len(ep_indices) <= 15 else None

            ax2_3d.plot(pos[:,0], pos[:,1], pos[:,2], color=color, lw=0.8, alpha=0.8, label=label)
            ax2_3d.scatter(*pos[0], color=color, s=20, zorder=5)

            ax2_xy.plot(pos[:,0]*100, pos[:,1]*100, color=color, lw=0.8, alpha=0.8, label=label)
            ax2_xy.scatter(pos[0,0]*100, pos[0,1]*100, color=color, s=15, zorder=5)

            ax2_xz.plot(pos[:,0]*100, pos[:,2]*100, color=color, lw=0.8, alpha=0.8)
            ax2_xz.scatter(pos[0,0]*100, pos[0,2]*100, color=color, s=15, zorder=5)

        set_axes_equal_3d(ax2_3d, all_pos_for_eq)

        ax2_3d.set_xlabel('X (m)'); ax2_3d.set_ylabel('Y (m)'); ax2_3d.set_zlabel('Z (m)')
        ax2_3d.set_title('3D TCP  (equal scale)')
        if len(ep_indices) <= 15:
            ax2_3d.legend(fontsize=6, loc='upper left')

        ax2_xy.set_xlabel('X (cm)'); ax2_xy.set_ylabel('Y (cm)')
        ax2_xy.set_title('Top-down XY  (equal scale)')
        ax2_xy.set_aspect('equal'); ax2_xy.grid(True, alpha=0.3)

        ax2_xz.set_xlabel('X (cm)'); ax2_xz.set_ylabel('Z (cm)')
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
