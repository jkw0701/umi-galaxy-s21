"""
Pose Graph Optimization (PGO) for DROID-SLAM trajectory refinement.

ArUco 마커 관측값을 절대 위치 제약(Prior Factor)으로 사용하여
DROID-SLAM 궤적의 누적 드리프트를 교정합니다.

의존 라이브러리:
    pip install gtsam            # factor graph optimization
    pip install scipy numpy pandas pickle-mixin

사용법:
    python scripts/pgo_refine_trajectory.py \\
        --csv   <slam_output>/camera_trajectory.csv \\
        --pkl   <slam_output>/tag_detection.pkl \\
        --json  <slam_output>/tx_slam_tag.json \\
        -o      <slam_output>/camera_trajectory_pgo.csv \\
        [--tag_id 13] [--max_center_dist 0.6] [--sigma_prior 0.02]

출력:
    camera_trajectory_pgo.csv  ← 교정된 궤적 (원본과 동일한 컬럼 구조)
    pgo_summary.png            ← 교정 전/후 비교 그래프

파이프라인 통합:
    06_generate_dataset_plan.py 의 --csv 인자에
    camera_trajectory_pgo.csv 를 넣으면 됩니다.
    (scale_factor는 tx_slam_tag.json 에서 그대로 읽음)
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import pickle
import json
import click
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from umi.common.pose_util import pose_to_mat

# ── GTSAM import (optional; falls back to linear interpolation) ──────────────
try:
    import gtsam
    from gtsam import (
        NonlinearFactorGraph, Values, Pose3, Rot3, Point3,
        PriorFactorPose3, BetweenFactorPose3,
        noiseModel, LevenbergMarquardtOptimizer, LevenbergMarquardtParams,
    )
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False
    print("[WARNING] gtsam not found. Falling back to linear elastic-band optimization.")
    print("          pip install gtsam  to enable full PGO.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_slam_csv(csv_path: pathlib.Path):
    """
    camera_trajectory.csv → (df, cam_pos [N,3], cam_rot_matrix [N,3,3], timestamps [N])
    """
    df = pd.read_csv(csv_path)
    is_valid = ~df['is_lost']
    df_v = df.loc[is_valid].reset_index(drop=True)

    pos = df_v[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    quat = df_v[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy(dtype=np.float64)
    rot_mat = Rotation.from_quat(quat).as_matrix()
    ts = df_v['timestamp'].to_numpy(dtype=np.float64)
    return df_v, pos, rot_mat, ts


def collect_aruco_anchors_tag_frame(
        df_v, pos_tag, rot_tag,
        tag_detection_results, ts,
        tag_id, image_width, image_height,
        max_center_dist,
        min_tag_dist=0.10,
        min_marker_pixels=80,
        min_squareness=0.7):
    """
    Tag frame에서 ArUco-only anchor를 수집.

    핵심:
        anchor = inv(tx_cam_tag)[:3,3]   ← 순수 ArUco, SLAM/tx_slam_tag 완전 무관
        compare = pos_tag[frame_idx]     ← SLAM이 tag frame에서 추정한 위치

    두 값의 차이 = tx_slam_tag 오차 + SLAM 드리프트 전부 포함.
    PGO가 이 오차를 직접 교정할 수 있음.

    Parameters
    ----------
    pos_tag  : [N, 3] SLAM 궤적, tag frame (metric)
    rot_tag  : [N, 3, 3] SLAM 회전, tag frame

    Returns
    -------
    anchors : list of dict (pos_metric은 tag frame 기준)
    """
    video_ts = np.array([x['time'] for x in tag_detection_results])
    img_center = np.array([image_width, image_height], dtype=np.float64) / 2

    anchors_raw = []

    for frame_idx in range(len(df_v)):
        t = ts[frame_idx]
        vid_idx = int(np.argmin(np.abs(video_ts - t)))
        td = tag_detection_results[vid_idx]
        if tag_id not in td.get('tag_dict', {}):
            continue

        tag = td['tag_dict'][tag_id]
        corners = tag['corners']  # (4, 2)
        tag_center_pix = corners.mean(axis=0)
        dist_center = np.linalg.norm(tag_center_pix - img_center) / img_center[1]
        if dist_center > max_center_dist:
            continue

        # ── 마커 픽셀 품질 필터 ──────────────────────────────────────────────
        side_lengths = np.array([
            np.linalg.norm(corners[1] - corners[0]),
            np.linalg.norm(corners[2] - corners[1]),
            np.linalg.norm(corners[3] - corners[2]),
            np.linalg.norm(corners[0] - corners[3]),
        ])
        if side_lengths.mean() < min_marker_pixels:
            continue
        if side_lengths.min() / side_lengths.max() < min_squareness:
            continue

        pose = np.concatenate([tag['tvec'], tag['rvec']])
        tx_cam_tag = pose_to_mat(pose)
        tag_dist = float(np.linalg.norm(tx_cam_tag[:3, 3]))
        if tag_dist < min_tag_dist or tag_dist > 2.0:
            continue

        # ── 카메라 위치를 tag frame에서 직접 계산 (SLAM/tx_slam_tag 완전 무관) ──
        # tx_cam_tag : T^cam_tag  (tag frame → camera frame 변환)
        # inv(tx_cam_tag) = T^tag_cam (camera frame → tag frame 변환)
        # [:3,3] : camera 원점의 tag frame 좌표
        cam_pos_in_tag = np.linalg.inv(tx_cam_tag)[:3, 3]

        anchors_raw.append({
            'frame_idx': frame_idx,
            'pos_metric': cam_pos_in_tag.copy(),       # ArUco-only anchor (tag frame)
            'pos_slam':   pos_tag[frame_idx].copy(),   # SLAM 추정 (tag frame, 비교용)
            'rot_mat':    rot_tag[frame_idx].copy(),
            'tag_dist':   tag_dist,
        })

    if len(anchors_raw) == 0:
        return []

    # 아웃라이어 제거 (90% quantile)
    anchor_positions = np.array([a['pos_metric'] for a in anchors_raw])
    median_pos = np.median(anchor_positions, axis=0)
    dists = np.linalg.norm(anchor_positions - median_pos, axis=1)
    thresh = np.quantile(dists, 0.9)
    anchors = [a for a, d in zip(anchors_raw, dists) if d <= thresh]

    if anchors:
        drift = np.array([np.linalg.norm(a['pos_metric'] - a['pos_slam'])
                          for a in anchors]) * 100
        print(f"  ArUco(tag frame) vs SLAM(tag frame) 오차:")
        print(f"  mean={drift.mean():.1f}cm  max={drift.max():.1f}cm  "
              f"p95={np.percentile(drift, 95):.1f}cm")
        print(f"  (이 값이 tx_slam_tag 오차 + SLAM 드리프트의 합)")

    print(f"ArUco anchor frames: {len(anchors_raw)} detected, "
          f"{len(anchors)} after outlier removal")
    return anchors


# ─────────────────────────────────────────────────────────────────────────────
# PGO backend 1: GTSAM
# ─────────────────────────────────────────────────────────────────────────────

def run_gtsam_pgo(pos_metric, rot_mat, anchors, sigma_odom, sigma_prior):
    """
    Factor graph:
      - BetweenFactorPose3 for consecutive frames  (odometry, σ=sigma_odom)
      - PriorFactorPose3   for ArUco-visible frames (σ=sigma_prior)
      - Strong prior on first pose to fix gauge freedom

    Returns
    -------
    refined_pos : np.ndarray [N, 3]
    """
    N = len(pos_metric)
    graph = NonlinearFactorGraph()
    init = Values()

    noise_odom  = noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01,  # rot (rad)
                                                        sigma_odom, sigma_odom, sigma_odom]))
    noise_prior = noiseModel.Diagonal.Sigmas(np.array([0.005, 0.005, 0.005,  # rot
                                                        sigma_prior, sigma_prior, sigma_prior]))
    # 첫 프레임: rotation만 고정, translation은 자유 (constant offset 보정 허용)
    # tag frame PGO에서 ArUco anchor가 절대 위치 기준을 제공하므로
    # 첫 프레임 translation을 고정하면 전체 이동이 막혀 경로가 구부러짐
    noise_first = noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6,   # rot: 고정
                                                        0.5,  0.5,  0.5]))  # trans: 자유

    def to_pose3(p, R):
        return Pose3(Rot3(R), Point3(*p))

    # Initial values & odometry edges
    for i in range(N):
        pose_i = to_pose3(pos_metric[i], rot_mat[i])
        init.insert(i, pose_i)

        if i > 0:
            pose_prev = to_pose3(pos_metric[i - 1], rot_mat[i - 1])
            rel = pose_prev.between(pose_i)
            graph.add(BetweenFactorPose3(i - 1, i, rel, noise_odom))

    # 첫 프레임 rotation 고정 (rotation gauge freedom 방지)
    graph.add(PriorFactorPose3(0, to_pose3(pos_metric[0], rot_mat[0]), noise_first))

    # ArUco prior factors
    anchor_set = {a['frame_idx'] for a in anchors}
    for a in anchors:
        fi = a['frame_idx']
        if fi >= N:
            continue
        prior_pose = to_pose3(a['pos_metric'], a['rot_mat'])
        graph.add(PriorFactorPose3(fi, prior_pose, noise_prior))

    print(f"  Factor graph: {N} nodes, "
          f"{N-1} odometry edges, {len(anchors)} prior factors")

    params = LevenbergMarquardtParams()
    params.setVerbosity('ERROR')
    optimizer = LevenbergMarquardtOptimizer(graph, init, params)
    result = optimizer.optimize()

    refined_pos = np.array([result.atPose3(i).translation() for i in range(N)])
    return refined_pos


# ─────────────────────────────────────────────────────────────────────────────
# PGO backend 2: Elastic-band (scipy fallback)
# ─────────────────────────────────────────────────────────────────────────────

def run_elastic_pgo(pos_metric, anchors, sigma_prior):
    """
    Simpler optimization using scipy (no rotation refinement).
    Minimizes:
      Σ_odom  ||p_i - p_{i-1} - delta_i||^2 / sigma_odom^2
    + Σ_anchor ||p_i - anchor_pos||^2 / sigma_prior^2

    where delta_i = original relative displacement (preserved as odometry).
    """
    from scipy.optimize import minimize

    N = len(pos_metric)
    orig_delta = np.diff(pos_metric, axis=0)  # [N-1, 3] original relative moves

    w_odom  = 1.0
    w_prior = (0.05 / sigma_prior) ** 2  # 5cm odom을 기준으로 prior weight 조정

    anchor_dict = {a['frame_idx']: a['pos_metric'] for a in anchors}

    x0 = pos_metric.flatten()

    def objective(x):
        p = x.reshape(N, 3)
        # odometry residuals (keep relative displacements close to original)
        odom_res = p[1:] - p[:-1] - orig_delta
        cost = w_odom * np.sum(odom_res ** 2)
        # anchor residuals
        for fi, apos in anchor_dict.items():
            if fi < N:
                cost += w_prior * np.sum((p[fi] - apos) ** 2)
        return cost

    def gradient(x):
        p = x.reshape(N, 3)
        grad = np.zeros_like(p)
        delta = p[1:] - p[:-1] - orig_delta
        # ∂cost/∂p_i from odom terms
        grad[1:]  += 2 * w_odom * delta
        grad[:-1] -= 2 * w_odom * delta
        # ∂cost/∂p_i from anchor terms
        for fi, apos in anchor_dict.items():
            if fi < N:
                grad[fi] += 2 * w_prior * (p[fi] - apos)
        return grad.flatten()

    print(f"  Elastic-band: {N} frames, {len(anchor_dict)} anchors")
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                   options={'maxiter': 2000, 'ftol': 1e-12})
    if not res.success:
        print(f"  [WARNING] Optimization did not converge: {res.message}")

    return res.x.reshape(N, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_plot(pos_orig, pos_refined, anchors, output_path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    anchor_frames = [a['frame_idx'] for a in anchors]
    anchor_pos    = np.array([a['pos_slam'] for a in anchors])   # SLAM 원래 위치

    # start-relative
    o0 = pos_orig[0]
    r0 = pos_refined[0]
    po = pos_orig    - o0
    pr = pos_refined - r0
    if len(anchor_pos) > 0:
        ap = anchor_pos - o0

    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('PGO Trajectory Refinement', fontsize=11)

    panels = [
        (fig.add_subplot(141, projection='3d'), None, None, '3D'),
        (fig.add_subplot(142), 0, 1, 'XY (top)'),
        (fig.add_subplot(143), 0, 2, 'XZ (side)'),
        (fig.add_subplot(144), 1, 2, 'YZ (front)'),
    ]

    labels = ['X (m)', 'Y (m)', 'Z (m)']

    for ax, xi, yi, title in panels:
        if title == '3D':
            ax.plot(po[:, 0], po[:, 1], po[:, 2],
                    color='#90CAF9', lw=1.0, alpha=0.8, label='Original SLAM')
            ax.plot(pr[:, 0], pr[:, 1], pr[:, 2],
                    color='#1976D2', lw=1.4, label='PGO refined')
            if len(anchor_pos) > 0:
                ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2],
                           color='#F44336', s=20, zorder=5, label='ArUco anchor')
            ax.set_xlabel('X', fontsize=7); ax.set_ylabel('Y', fontsize=7)
            ax.set_zlabel('Z', fontsize=7); ax.tick_params(labelsize=6)
        else:
            ax.plot(po[:, xi], po[:, yi],
                    color='#90CAF9', lw=1.0, alpha=0.8, label='Original SLAM')
            ax.plot(pr[:, xi], pr[:, yi],
                    color='#1976D2', lw=1.4, label='PGO refined')
            if len(anchor_pos) > 0:
                ax.scatter(ap[:, xi], ap[:, yi],
                           color='#F44336', s=20, zorder=5)
            ax.set_xlabel(labels[xi], fontsize=7)
            ax.set_ylabel(labels[yi], fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.25)
            ax.set_aspect('equal', adjustable='datalim')

        ax.set_title(title, fontsize=9)
        if title == '3D':
            ax.legend(fontsize=6)

    # Correction magnitude over frames
    diff = np.linalg.norm(pos_refined - pos_orig, axis=1) * 100
    inset = fig.add_axes([0.76, 0.12, 0.22, 0.30])
    inset.plot(diff, color='#7B1FA2', lw=1.0)
    inset.axhline(diff.mean(), color='gray', ls='--', lw=0.8,
                  label=f'mean {diff.mean():.1f}cm')
    if anchor_frames:
        for af in anchor_frames:
            inset.axvline(af, color='#F44336', lw=0.5, alpha=0.5)
    inset.set_xlabel('frame', fontsize=7)
    inset.set_ylabel('correction (cm)', fontsize=7)
    inset.tick_params(labelsize=6)
    inset.legend(fontsize=6)
    inset.grid(True, alpha=0.2)
    inset.set_title('PGO correction', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison plot: {output_path}")


def save_debug_plot(pos_orig, pos_refined, anchors, output_path):
    """
    축별(X/Y/Z) SLAM vs PGO 위치 + 마커 가시성 + 보정량 디버그 플롯.

    각 서브플롯:
      - 상단 3개: X, Y, Z 축별 위치 (SLAM 원본 vs PGO 보정)
          · 빨간 세로선: ArUco 마커가 보인 프레임
          · ×  : SLAM이 말하는 카메라 위치 (anchor pos_slam)
          · ●  : 태그 고정 조건으로 계산한 기대 위치 (anchor pos_metric)
          · ×와 ● 차이 = SLAM이 해당 프레임에서 얼마나 틀렸는지
      - 하단: 축별 PGO 보정량 ΔX, ΔY, ΔZ + 총 보정 크기
    """
    import matplotlib.pyplot as plt

    N = len(pos_orig)
    frames = np.arange(N)

    anchor_frames      = [a['frame_idx'] for a in anchors]
    anchor_pos_slam    = np.array([a['pos_slam']   for a in anchors]) * 100  # cm
    anchor_pos_expect  = np.array([a['pos_metric'] for a in anchors]) * 100  # cm

    correction = (pos_refined - pos_orig) * 100  # cm [N, 3]
    total_corr = np.linalg.norm(correction, axis=1)

    axis_labels   = ['X', 'Y', 'Z']
    col_orig      = ['#90CAF9', '#A5D6A7', '#FFCC80']
    col_refined   = ['#1565C0', '#2E7D32', '#E65100']

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    fig.suptitle('PGO Debug: 축별 보정량 & ArUco 마커 가시성', fontsize=12)

    # ── 축별 위치 플롯 ────────────────────────────────────────────────────────
    for i in range(3):
        ax = axes[i]
        orig_cm    = pos_orig[:, i]    * 100
        refined_cm = pos_refined[:, i] * 100

        ax.plot(frames, orig_cm,    color=col_orig[i],    lw=1.0, alpha=0.7,
                label=f'SLAM {axis_labels[i]}')
        ax.plot(frames, refined_cm, color=col_refined[i], lw=1.3,
                label=f'PGO {axis_labels[i]}')

        # 마커 가시 프레임 → 배경 음영
        for af in anchor_frames:
            ax.axvline(af, color='#F44336', lw=0.6, alpha=0.35)

        # SLAM 위치 (×) vs 태그 기반 기대 위치 (●)
        if len(anchor_frames) > 0:
            ax.scatter(anchor_frames, anchor_pos_slam[:, i],
                       marker='x', color='#757575', s=30, lw=1.2, zorder=5,
                       label='Anchor (SLAM pos)')
            ax.scatter(anchor_frames, anchor_pos_expect[:, i],
                       marker='o', color='#F44336', s=18, zorder=5,
                       label='Anchor (expected)')
            # SLAM vs expected를 화살표로 연결
            for af, ys, ye in zip(anchor_frames,
                                   anchor_pos_slam[:, i],
                                   anchor_pos_expect[:, i]):
                ax.annotate('', xy=(af, ye), xytext=(af, ys),
                            arrowprops=dict(arrowstyle='->', color='#F44336',
                                            lw=0.8, alpha=0.6))

        ax.set_ylabel(f'{axis_labels[i]} (cm)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.2)

    # ── 하단: 축별 + 총 보정량 ──────────────────────────────────────────────
    ax4 = axes[3]
    for i in range(3):
        ax4.plot(frames, correction[:, i], color=col_refined[i],
                 lw=1.0, alpha=0.85, label=f'Δ{axis_labels[i]}')
    ax4.plot(frames, total_corr, color='black', lw=1.4, ls='--',
             label=f'|Δtotal|  mean={total_corr.mean():.2f}cm')
    ax4.axhline(0, color='gray', lw=0.7)

    for af in anchor_frames:
        ax4.axvline(af, color='#F44336', lw=0.6, alpha=0.35)

    # anchor 위치에서의 오차 크기 표시
    if len(anchor_frames) > 0:
        anchor_err = np.linalg.norm(anchor_pos_expect - anchor_pos_slam, axis=1)
        ax4.scatter(anchor_frames, np.zeros(len(anchor_frames)),
                    marker='^', color='#F44336', s=30, zorder=6,
                    label='Anchor frame')
        for af, err in zip(anchor_frames, anchor_err):
            ax4.annotate(f'{err:.1f}', (af, 0.3),
                         fontsize=5, color='#B71C1C', ha='center')

    ax4.set_ylabel('Correction (cm)', fontsize=9)
    ax4.set_xlabel('Frame index', fontsize=9)
    ax4.legend(fontsize=7, loc='upper right', ncol=3)
    ax4.grid(True, alpha=0.2)

    # 마커 가시 구간 설명
    if anchor_frames:
        fig.text(0.01, 0.01,
                 f'빨간 세로선 = ArUco 마커 가시 ({len(anchor_frames)}개 프레임)  |  '
                 f'× = SLAM 위치  ● = 태그 기반 기대 위치  화살표 = 드리프트 방향·크기',
                 fontsize=7, color='#555555')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved debug plot: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option('--csv',   'csv_path',  required=True,
              help='DROID-SLAM camera_trajectory.csv')
@click.option('--pkl',   'pkl_path',  required=True,
              help='tag_detection.pkl (from detect_aruco.py)')
@click.option('--json',  'json_path', required=True,
              help='tx_slam_tag.json (from calibrate_slam_tag.py)')
@click.option('-o', '--output', default=None,
              help='Output CSV path (default: <csv_dir>/camera_trajectory_pgo.csv)')
@click.option('--tag_id', type=int, default=13, show_default=True)
@click.option('--image_width',  type=int, default=2704, show_default=True)
@click.option('--image_height', type=int, default=2028, show_default=True)
@click.option('--max_center_dist', type=float, default=0.9, show_default=True,
              help='ArUco detection filter: max normalized dist from image center')
@click.option('--sigma_prior', type=float, default=0.02, show_default=True,
              help='ArUco anchor noise std (m). 0.02 = 2cm. Smaller = tighter anchor.')
@click.option('--sigma_odom',  type=float, default=0.005, show_default=True,
              help='Odometry noise std (m). GTSAM only.')
@click.option('--min_tag_dist', type=float, default=0.10, show_default=True,
              help='ArUco anchor 최소 거리 (m). 기본 0.10m. '
                   '테이블 근접 구간 포함하려면 낮추기 (최소 0.05 권장).')
@click.option('--min_marker_pixels', type=float, default=80, show_default=True,
              help='마커 평균 변 길이 최솟값 (픽셀). 클수록 엄격 (고품질 관측만 사용). '
                   '기본 80px. 높이면 anchor 수 줄고 품질 올라감.')
@click.option('--min_squareness', type=float, default=0.7, show_default=True,
              help='마커 정사각형 비율 최솟값 (0~1). min변/max변. '
                   '기본 0.7. 높이면 비스듬한 관측 제외.')
@click.option('--no_plot', is_flag=True, default=False)
def main(csv_path, pkl_path, json_path, output,
         tag_id, image_width, image_height, max_center_dist,
         sigma_prior, sigma_odom, min_tag_dist,
         min_marker_pixels, min_squareness, no_plot):

    csv_path  = pathlib.Path(csv_path)
    pkl_path  = pathlib.Path(pkl_path)
    json_path = pathlib.Path(json_path)

    if output is None:
        output = csv_path.parent / 'camera_trajectory_pgo.csv'
    output = pathlib.Path(output)

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"Loading SLAM trajectory: {csv_path}")
    df_v, pos_droid, rot_mat, ts = load_slam_csv(csv_path)

    tag_data = json.load(open(json_path))
    scale_factor = float(tag_data.get('scale_factor', 1.0))
    tx_slam_tag_mat = np.array(tag_data['tx_slam_tag'], dtype=np.float64)
    print(f"Scale factor (from tx_slam_tag.json): {scale_factor:.4f}")

    pos_metric = pos_droid / scale_factor
    print(f"  Trajectory: {len(pos_metric)} valid frames")

    tag_detection_results = pickle.load(open(pkl_path, 'rb'))

    # ── SLAM 궤적을 tag frame으로 변환 ─────────────────────────────────────
    # tx_slam_tag_mat : T^slam_tag (tag frame → SLAM metric frame)
    # tx_tag_slam     : T^tag_slam (SLAM metric frame → tag frame)
    tx_tag_slam = np.linalg.inv(tx_slam_tag_mat)
    R_tag_slam  = tx_tag_slam[:3, :3]

    # pos_tag[i] = R_tag_slam @ pos_metric[i] + t_tag_slam
    ones = np.ones((len(pos_metric), 1))
    pos_tag = (tx_tag_slam @ np.hstack([pos_metric, ones]).T).T[:, :3]  # (N, 3)

    # rot_tag[i] = R_tag_slam @ rot_mat[i]  (카메라 회전을 tag frame으로 표현)
    rot_tag = np.array([R_tag_slam @ r for r in rot_mat])  # (N, 3, 3)

    print(f"  Trajectory in tag frame: range "
          f"X={pos_tag[:,0].ptp()*100:.1f}cm  "
          f"Y={pos_tag[:,1].ptp()*100:.1f}cm  "
          f"Z={pos_tag[:,2].ptp()*100:.1f}cm")

    # ── Collect ArUco anchors (tag frame, SLAM 완전 무관) ──────────────────
    anchors = collect_aruco_anchors_tag_frame(
        df_v, pos_tag, rot_tag, tag_detection_results, ts,
        tag_id, image_width, image_height, max_center_dist,
        min_tag_dist=min_tag_dist,
        min_marker_pixels=min_marker_pixels,
        min_squareness=min_squareness,
    )

    if len(anchors) == 0:
        print("[WARNING] No ArUco anchors found. Saving original trajectory.")
        df_out = df_v.copy()
        df_out.to_csv(output, index=False)
        print(f"Saved: {output}")
        return

    # ── PGO (tag frame에서 실행) ──────────────────────────────────────────
    print(f"\nRunning PGO in TAG FRAME ({'GTSAM' if HAS_GTSAM else 'elastic-band'})...")
    if HAS_GTSAM:
        refined_pos_tag = run_gtsam_pgo(pos_tag, rot_tag, anchors, sigma_odom, sigma_prior)
    else:
        refined_pos_tag = run_elastic_pgo(pos_tag, anchors, sigma_prior)

    # ── Statistics ──────────────────────────────────────────────────────────
    diff = np.linalg.norm(refined_pos_tag - pos_tag, axis=1) * 100
    print(f"\nPGO correction summary (tag frame):")
    print(f"  mean correction : {diff.mean():.2f} cm")
    print(f"  max  correction : {diff.max():.2f} cm")
    print(f"  p95  correction : {np.percentile(diff, 95):.2f} cm")

    res_list = [np.linalg.norm(refined_pos_tag[a['frame_idx']] - a['pos_metric']) * 100
                for a in anchors if a['frame_idx'] < len(refined_pos_tag)]
    if res_list:
        print(f"  anchor residual (post-PGO): mean={np.mean(res_list):.2f}cm  "
              f"max={np.max(res_list):.2f}cm")

    # ── Save: tag frame 결과를 DROID 단위로 back-encode ─────────────────────
    # 06이 하는 변환: p_tag = tx_tag_slam @ (p_droid / scale_factor)
    # 역산: p_droid = scale_factor * tx_slam_tag_mat @ [p_tag, 1]
    ones_r = np.ones((len(refined_pos_tag), 1))
    refined_slam_metric = (tx_slam_tag_mat @ np.hstack([refined_pos_tag, ones_r]).T).T[:, :3]
    refined_droid = refined_slam_metric * scale_factor

    df_out = df_v.copy()
    df_out['x'] = refined_droid[:, 0]
    df_out['y'] = refined_droid[:, 1]
    df_out['z'] = refined_droid[:, 2]
    df_out.to_csv(output, index=False)
    print(f"\nSaved refined trajectory: {output}")
    print(f"  (tag frame PGO → back-encoded to DROID units, scale_factor={scale_factor:.4f})")

    # ── Plot (tag frame 기준) ─────────────────────────────────────────────
    if not no_plot:
        plot_path = output.with_name('pgo_comparison.png')
        save_comparison_plot(pos_tag, refined_pos_tag, anchors, plot_path)

        debug_path = output.with_name('pgo_debug.png')
        save_debug_plot(pos_tag, refined_pos_tag, anchors, debug_path)


if __name__ == '__main__':
    main()
