"""
Per-demo calibration for the no-SLAM pipeline (Galaxy S21).

Each ARCore recording has its own world frame, so we compute tx_slam_tag
per-demo. Two modes:

  [default] --ref first_pose
    Use the first valid ARCore pose as the episode reference frame.
    No ArUco table tags required. The policy learns trajectories
    relative to the gripper's starting pose.

  --ref aruco
    Run calibrate_slam_tag.py using ArUco table tags.
    Requires the table ArUco tag to be visible in every demo.
    All demos are expressed in the table frame.

Coordinate frame note:
  ARCore world frame is Y-up (gravity = -Y).
  UMI/GoPro pipeline expects Z-up (gravity = -Z).

  To keep the dataset consistent with the GoPro pipeline and the robot
  deployment code, tx_slam_tag is defined so that the resulting "tag frame"
  is Z-up.  This is done by applying R_arcore_to_zup = Rx(-90°):

      ARCore:  X=right,  Y=up,      Z=backward (away from scene)
      Z-up:    X=right,  Y=forward, Z=up

  Rotation matrix (columns = Z-up axes expressed in ARCore frame):
      R = Rx(+90°) = [[1,  0,  0],
                      [0,  0, -1],
                      [0,  1,  0]]

  After this fix the dataset EEF Z-axis corresponds to physical height,
  matching the GoPro-based datasets and the robot deployment code.

Gripper range calibration (gripper ArUco tags) is always performed.

Usage:
    python droid_slam_s21/05_run_calibrations_no_slam.py <session_dir>
    python droid_slam_s21/05_run_calibrations_no_slam.py --ref aruco <session_dir>
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import json
import pickle
import tempfile
import shutil
import click
import subprocess
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation


def _patch_aruco_config_for_slam_tag(tag_pkl_path: pathlib.Path,
                                      tag_id: int, marker_size_m: float,
                                      cmd: list):
    """
    tag_detection.pkl 옆의 aruco_config 정보를 읽어
    marker_size를 덮어쓴 임시 yaml을 만들고 cmd에 --aruco_yaml 인자를 추가.

    calibrate_slam_tag.py는 tag_detection.pkl 안의 tvec을 사용하는데,
    tvec은 이미 detect 시점의 marker_size로 계산됨.
    그래서 여기서 marker_size를 바꿔봤자 tvec에는 영향이 없음.
    → 대신 calibrate_slam_tag.py가 tag_id의 tvec만 사용하므로,
      marker_size는 이미 tag_detection.pkl 생성 시 반영됨.
    이 함수는 re-detection 없이 크기가 맞는지 경고만 출력.
    """
    # tag_detection.pkl에서 첫 번째 tvec 크기로 실제 감지된 거리 추정
    try:
        tdr = pickle.load(open(tag_pkl_path, 'rb'))
        for frame in tdr:
            if tag_id in frame.get('tag_dict', {}):
                tvec = frame['tag_dict'][tag_id]['tvec']
                dist = float(np.linalg.norm(tvec))
                if abs(dist - marker_size_m * 2) > marker_size_m:
                    print(f"  Info: tag {tag_id} detected at ~{dist*100:.1f}cm distance. "
                          f"Expected marker_size={marker_size_m*100:.1f}cm.")
                break
    except Exception:
        pass


def make_tx_slam_tag_from_first_pose(csv_path: pathlib.Path) -> np.ndarray:
    """
    Build a 4x4 tx_slam_tag matrix that defines a Z-up reference frame
    anchored at the first DROID-SLAM pose.

    DROID-SLAM coordinate frame (monocular, S21 ultrawide looking down at table):
        X: camera right
        Y: camera image-down direction
        Z: camera optical axis = physical DOWN (toward table)

    UMI/robot expects Z-up tag frame (gravity = -Z).

    Empirically verified mapping DROID → tag(Z-up):
        X_tag = +DROID X  (right, unchanged)
        Y_tag = -DROID Y  (image-down → tag backward)
        Z_tag = -DROID Z  (physical down → tag -Z = down ✓)

    Verification: camera moves DOWN → DROID Z increases →
    tag Z decreases (negative = lower height) ✓
    """
    df = pd.read_csv(csv_path)
    valid = df[df['is_lost'] == 0]
    if len(valid) == 0:
        raise ValueError(f"No valid (non-lost) poses in {csv_path}")

    first = valid.iloc[0]
    pos = np.array([first['x'], first['y'], first['z']])

    # Empirically verified from S21 ultrawide pick motion data:
    #   Camera moves DOWN toward table → DROID Z increases (+)
    #   Camera moves UP away from table → DROID Z decreases (-)
    #   DROID Y is roughly camera-image-down direction
    #
    # Desired mapping to Z-up tag frame:
    #   DROID X → tag +X  (right, unchanged)
    #   DROID Y → tag -Y  (image-down → tag backward)
    #   DROID Z → tag -Z  (physical down → tag -Z = down ✓)
    #
    # R_droid_to_zup (each row = tag axis dot product with DROID axes):
    R_droid_to_zup = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=float)

    tx = np.eye(4, dtype=float)
    tx[:3, :3] = R_droid_to_zup
    tx[:3, 3] = pos
    return tx


@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('--ref', type=click.Choice(['first_pose', 'aruco']), default='first_pose',
              show_default=True,
              help=(
                  'first_pose: 첫 번째 유효한 DROID-SLAM 포즈를 원점으로 사용 (ArUco 테이블 태그 불필요). '
                  'aruco: 테이블에 붙인 ArUco 마커로 scale 보정 (데모 영상에 마커가 보여야 함).'
              ))
@click.option('--table_tag_id', type=int, default=13, show_default=True,
              help='[--ref aruco 전용] 테이블에 붙인 ArUco 마커 ID.')
@click.option('--table_marker_size', type=float, default=None,
              help='[--ref aruco 전용] 마커 실제 크기 (미터). '
                   '지정하면 aruco_config.yaml의 값을 덮어씁니다.')
@click.option('--aruco_config', default=None,
              help='[--ref aruco 전용] aruco_config.yaml 경로. '
                   '기본: <session_dir>/../demos/../ 에서 자동 탐색.')
@click.option('--ignore_gripper_tags', is_flag=True, default=False,
              help='그리퍼 ArUco 태그 캘리브레이션을 건너뜁니다.')
@click.option('--pgo', is_flag=True, default=False,
              help='[--ref aruco 전용] Pose Graph Optimization으로 궤적 드리프트 교정. '
                   'camera_trajectory_pgo.csv를 생성하고 dataset 생성에 사용.')
@click.option('--sigma_prior', type=float, default=0.02, show_default=True,
              help='PGO: ArUco anchor noise std (m). 작을수록 마커 위치에 강하게 고정.')
@click.option('--sigma_odom', type=float, default=0.005, show_default=True,
              help='PGO: Odometry noise std (m).')
@click.option('--max_center_dist', type=float, default=0.9, show_default=True,
              help='PGO: ArUco detection filter (normalized dist from image center).')
@click.option('--min_tag_dist', type=float, default=0.10, show_default=True,
              help='PGO: ArUco anchor 최소 거리 (m). 낮출수록 테이블 근접 구간도 포함.')
@click.option('--min_marker_pixels', type=float, default=80, show_default=True,
              help='PGO: 마커 최소 픽셀 크기. 클수록 고품질 관측만 anchor로 사용.')
@click.option('--min_squareness', type=float, default=0.7, show_default=True,
              help='PGO: 마커 정사각형 비율 최솟값 (0~1). 비스듬한 관측 제외.')
def main(session_dir, ref, table_tag_id, table_marker_size, aruco_config, ignore_gripper_tags,
         pgo, sigma_prior, sigma_odom, max_center_dist, min_tag_dist,
         min_marker_pixels, min_squareness):
    script_dir = pathlib.Path(__file__).parent.parent / 'scripts'
    calibrate_gripper_range = script_dir / 'calibrate_gripper_range.py'
    assert calibrate_gripper_range.is_file(), f"Not found: {calibrate_gripper_range}"

    if ref == 'aruco':
        calibrate_slam_tag = script_dir / 'calibrate_slam_tag.py'
        assert calibrate_slam_tag.is_file(), f"Not found: {calibrate_slam_tag}"

    if pgo and ref != 'aruco':
        print("Warning: --pgo is only available with --ref aruco. Ignoring --pgo.")
        pgo = False

    if pgo:
        pgo_script = script_dir / 'pgo_refine_trajectory.py'
        assert pgo_script.is_file(), f"Not found: {pgo_script}"
        print(f"[PGO] sigma_prior={sigma_prior}m  sigma_odom={sigma_odom}m  "
              f"max_center_dist={max_center_dist}")

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        demos_dir = session / 'demos'

        for demo_dir in sorted(demos_dir.glob('demo_*/')):
            csv_path = demo_dir / 'camera_trajectory.csv'
            tag_path = demo_dir / 'tag_detection.pkl'
            slam_tag_path = demo_dir / 'tx_slam_tag.json'

            if slam_tag_path.is_file():
                print(f"Skipping {demo_dir.name}, tx_slam_tag.json already exists.")
                continue

            if not csv_path.is_file():
                print(f"Warning: no camera_trajectory.csv in {demo_dir.name}, skipping.")
                continue

            if ref == 'first_pose':
                try:
                    tx = make_tx_slam_tag_from_first_pose(csv_path)
                    json.dump({'tx_slam_tag': tx.tolist()}, open(slam_tag_path, 'w'))
                    print(f"  Written tx_slam_tag.json (first_pose) for {demo_dir.name}")
                except ValueError as e:
                    print(f"  Warning: {e}, skipping {demo_dir.name}")

            else:  # aruco
                if not tag_path.is_file():
                    print(f"Warning: no tag_detection.pkl in {demo_dir.name}, skipping.")
                    continue

                cmd = [
                    'python', str(calibrate_slam_tag),
                    '--tag_detection', str(tag_path),
                    '--csv_trajectory', str(csv_path),
                    '--output', str(slam_tag_path),
                    '--tag_id', str(table_tag_id),
                    '--image_width', '1920',
                    '--image_height', '1080',
                    '--max_center_dist', '2.0',
                    '--min_tag_dist', '0.1',
                ]

                if table_marker_size is not None:
                    _patch_aruco_config_for_slam_tag(
                        tag_path, table_tag_id, table_marker_size, cmd)

                result = subprocess.run(cmd, capture_output=False)
                if result.returncode != 0:
                    print(f"  Warning: calibrate_slam_tag failed for {demo_dir.name}")
                    continue

                # ── Optional PGO refinement ─────────────────────────────
                if pgo:
                    pgo_csv_path = demo_dir / 'camera_trajectory_pgo.csv'
                    cmd_pgo = [
                        'python', str(pgo_script),
                        '--csv',  str(csv_path),
                        '--pkl',  str(tag_path),
                        '--json', str(slam_tag_path),
                        '-o',     str(pgo_csv_path),
                        '--tag_id',          str(table_tag_id),
                        '--sigma_prior',     str(sigma_prior),
                        '--sigma_odom',      str(sigma_odom),
                        '--max_center_dist', str(max_center_dist),
                        '--min_tag_dist',       str(min_tag_dist),
                        '--min_marker_pixels',  str(min_marker_pixels),
                        '--min_squareness',     str(min_squareness),
                        '--image_width', '1920',
                        '--image_height', '1080',
                    ]
                    result_pgo = subprocess.run(cmd_pgo)
                    if result_pgo.returncode == 0:
                        print(f"  [PGO] Saved camera_trajectory_pgo.csv for {demo_dir.name}")
                    else:
                        print(f"  [PGO] Warning: PGO failed for {demo_dir.name}")

        # Gripper range calibration
        if ignore_gripper_tags:
            print("  Skipping gripper range calibration (--ignore_gripper_tags)")
            continue
        for gripper_dir in sorted(demos_dir.glob('gripper_calibration*/')):
            tag_path = gripper_dir / 'tag_detection.pkl'
            if not tag_path.is_file():
                print(f"Warning: no tag_detection.pkl in {gripper_dir.name}, skipping.")
                continue
            gripper_range_path = gripper_dir / 'gripper_range.json'
            cmd = [
                'python', str(calibrate_gripper_range),
                '--input', str(tag_path),
                '--output', str(gripper_range_path),
                '--tag_det_threshold', '0.7',
                '--nominal_z', '0.11',
            ]
            subprocess.run(cmd)


if __name__ == '__main__':
    main()
