"""
UMI SLAM Pipeline — Galaxy S21 version.

Two-phase workflow:
  [Phase 1] Calibration (once per device):
      python run_slam_pipeline_s21.py calibrate --video <checkerboard.mp4>

  [Phase 2] Process data (per session):
      DROID-SLAM (0.5x 영상):
          python run_slam_pipeline_s21.py process-droid <session_dir>

      ARCore (1.0x 영상, ar_pose 내장):
          python run_slam_pipeline_s21.py process-arcore --ref aruco <session_dir>

Calibration output is saved to example/calibration_s21/ by default.
The process command auto-detects calibration files from there.
"""

import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess


def get_default_calibration_dir():
    return pathlib.Path(ROOT_DIR) / 'example' / 'calibration_s21'


@click.group()
def cli():
    """UMI SLAM Pipeline for Galaxy S21."""
    pass


# ======================================================================
#  Phase 1: calibrate
# ======================================================================
@cli.command()
@click.option('--video', required=True, help='Checkerboard calibration video from S21')
@click.option('--output_dir', default=None, help='Calibration output dir (default: example/calibration_s21)')
@click.option('--checker_cols', type=int, default=9, help='Inner corners per row')
@click.option('--checker_rows', type=int, default=6, help='Inner corners per column')
@click.option('--square_size', type=float, default=0.025, help='Square size in meters')
@click.option('--fps', type=float, default=30.0, help='S21 recording FPS')
def calibrate(video, output_dir, checker_cols, checker_rows, square_size, fps):
    """[Phase 1] Run camera calibration from a checkerboard video. One-time setup."""
    script_path = pathlib.Path(ROOT_DIR) / 'scripts_slam_s21' / 'setup_s21_calibration.py'
    assert script_path.is_file(), f"Not found: {script_path}"

    if output_dir is None:
        output_dir = str(get_default_calibration_dir())

    cmd = [
        'python', str(script_path),
        '--video', video,
        '--output_dir', output_dir,
        '--checker_cols', str(checker_cols),
        '--checker_rows', str(checker_rows),
        '--square_size', str(square_size),
        '--fps', str(fps),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Calibration failed!")
        exit(1)

    print()
    print("Next step:")
    print(f"  python run_slam_pipeline_s21.py process <session_dir>")


# ======================================================================
#  Phase 2 (DROID-SLAM): process_droid
# ======================================================================
@cli.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None,
              help='Calibration directory (default: example/calibration_s21)')
@click.option('--ref', type=click.Choice(['first_pose', 'aruco']), default='first_pose',
              show_default=True,
              help='first_pose: use first valid pose as reference (no ArUco table tags needed). '
                   'aruco: require ArUco table tag visible in every demo.')
@click.option('--ignore_gripper_tags', is_flag=True, default=False,
              help='Skip gripper tag detection and assume gripper is always fully open.')
@click.option('--stride', type=int, default=1,
              help='Frame stride for DROID-SLAM (1=all frames, 2=every other frame)')
@click.option('--weights', type=str, default=None,
              help='Path to droid.pth (default: ../DROID-SLAM/droid.pth)')
@click.option('--no_mask', is_flag=True, default=False,
              help='Disable gripper mask for DROID-SLAM (use when gripper mask covers workspace features).')
@click.option('--pgo', is_flag=True, default=False,
              help='[--ref aruco 전용] Pose Graph Optimization으로 궤적 드리프트 교정. '
                   'camera_trajectory_pgo.csv를 생성하고 dataset 생성에 사용.')
@click.option('--sigma_prior', type=float, default=0.02, show_default=True,
              help='PGO ArUco anchor noise std (m). 작을수록 마커 위치에 강하게 고정.')
@click.option('--sigma_odom', type=float, default=0.005, show_default=True,
              help='PGO odometry noise std (m).')
@click.option('--max_center_dist', type=float, default=0.9, show_default=True,
              help='PGO ArUco filter: 이미지 중심으로부터 최대 정규화 거리.')
@click.option('--min_tag_dist', type=float, default=0.10, show_default=True,
              help='PGO ArUco anchor 최소 거리 (m). 낮출수록 테이블 근접 구간도 포함.')
def process_droid(session_dir, calibration_dir, ref, ignore_gripper_tags, stride, weights, no_mask,
                  pgo, sigma_prior, sigma_odom, max_center_dist, min_tag_dist):
    """[Phase 2 - DROID-SLAM] Build dataset using DROID-SLAM (no ORB-SLAM3, no IMU needed)."""
    script_dir = pathlib.Path(ROOT_DIR) / 'droid_slam_s21'

    if calibration_dir is None:
        calibration_dir = get_default_calibration_dir()
    else:
        calibration_dir = pathlib.Path(calibration_dir)

    intrinsics_path = calibration_dir / 's21_intrinsics_1080p.json'
    aruco_config_path = calibration_dir / 'aruco_config.yaml'

    if not intrinsics_path.is_file():
        print("=" * 60)
        print("  Calibration files not found!")
        print(f"  Expected in: {calibration_dir}")
        print()
        print("  Run calibration first:")
        print("    python run_slam_pipeline_s21.py calibrate --video <checkerboard.mp4>")
        print("=" * 60)
        exit(1)

    assert aruco_config_path.is_file(), \
        f"aruco_config.yaml not found in {calibration_dir}"

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        print(f"\nProcessing session (DROID-SLAM): {session.name}")
        print("=" * 60)

        # Step 00: Process videos
        _run_step(script_dir, "00_process_videos.py",
                  "00 Process Videos", [str(session)])

        # Step 01: DROID-SLAM (video → camera_trajectory.csv)
        droid_args = [
            '--input_dir', str(session / 'demos'),
            '--intrinsics', str(intrinsics_path),
            '--stride', str(stride),
        ]
        if weights:
            droid_args.extend(['--weights', weights])
        if no_mask:
            droid_args.append('--no_mask')
        _run_step(script_dir, "03_batch_slam.py",
                  "01 DROID-SLAM", droid_args)

        # Step 02: Detect ArUco
        _run_step(script_dir, "04_detect_aruco.py",
                  "02 Detect ArUco",
                  ['--input_dir', str(session / 'demos'),
                   '--camera_intrinsics', str(intrinsics_path),
                   '--aruco_yaml', str(aruco_config_path)])

        # Step 03: Per-demo calibration (+ optional PGO)
        calib_args = ['--ref', ref, str(session)]
        if ignore_gripper_tags:
            calib_args.append('--ignore_gripper_tags')
        if pgo:
            calib_args += [
                '--pgo',
                '--sigma_prior',     str(sigma_prior),
                '--sigma_odom',      str(sigma_odom),
                '--max_center_dist', str(max_center_dist),
                '--min_tag_dist',    str(min_tag_dist),
            ]
        _run_step(script_dir, "05_run_calibrations_per_episode.py",
                  "03 Run Calibrations (DROID-SLAM)", calib_args)

        # Step 04: Generate dataset plan
        dataset_plan_args = ['--input', str(session)]
        if ignore_gripper_tags:
            dataset_plan_args.append('--ignore_gripper_tags')
        if pgo:
            dataset_plan_args.append('--use_pgo')
        _run_step(script_dir, "06_generate_dataset_plan.py",
                  "04 Generate Dataset Plan (DROID-SLAM)",
                  dataset_plan_args)

        print()
        print("=" * 60)
        print(f"  Pipeline complete for: {session.name}")
        print(f"  Output: {session / 'dataset_plan.pkl'}")
        print("=" * 60)


# ======================================================================
#  Phase 2 (ARCore): process_arcore
# ======================================================================
@cli.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None,
              help='Calibration directory (default: example/calibration_s21)')
@click.option('--ref', type=click.Choice(['first_pose', 'aruco']), default='first_pose',
              show_default=True,
              help='first_pose: 첫 번째 유효한 ARCore 포즈를 원점으로 사용 (ArUco 테이블 태그 불필요). '
                   'aruco: 테이블 ArUco 마커 기준으로 좌표계 정렬 (데모 영상에 마커가 보여야 함).')
@click.option('--ignore_gripper_tags', is_flag=True, default=False,
              help='Skip gripper tag detection and assume gripper is always fully open.')
@click.option('--pgo', is_flag=True, default=False,
              help='[--ref aruco 전용] Pose Graph Optimization으로 궤적 드리프트 교정.')
@click.option('--sigma_prior', type=float, default=0.02, show_default=True,
              help='PGO ArUco anchor noise std (m).')
@click.option('--sigma_odom', type=float, default=0.005, show_default=True,
              help='PGO odometry noise std (m).')
@click.option('--max_center_dist', type=float, default=0.9, show_default=True,
              help='PGO ArUco filter: 이미지 중심으로부터 최대 정규화 거리.')
@click.option('--min_tag_dist', type=float, default=0.10, show_default=True,
              help='PGO ArUco anchor 최소 거리 (m).')
def process_arcore(session_dir, calibration_dir, ref, ignore_gripper_tags,
                   pgo, sigma_prior, sigma_odom, max_center_dist, min_tag_dist):
    """[Phase 2 - ARCore] Build dataset using ARCore poses from sensor_data.jsonl (no DROID-SLAM needed)."""
    script_dir = pathlib.Path(ROOT_DIR) / 'droid_slam_s21'

    if calibration_dir is None:
        calibration_dir = get_default_calibration_dir()
    else:
        calibration_dir = pathlib.Path(calibration_dir)

    intrinsics_path = calibration_dir / 's21_intrinsics_1080p.json'
    aruco_config_path = calibration_dir / 'aruco_config.yaml'

    if not intrinsics_path.is_file():
        print("=" * 60)
        print("  Calibration files not found!")
        print(f"  Expected in: {calibration_dir}")
        print()
        print("  Run calibration first:")
        print("    python run_slam_pipeline_s21.py calibrate --video <checkerboard.mp4>")
        print("=" * 60)
        exit(1)

    assert aruco_config_path.is_file(), \
        f"aruco_config.yaml not found in {calibration_dir}"

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        print(f"\nProcessing session (ARCore): {session.name}")
        print("=" * 60)

        # Step 00: Process videos
        _run_step(script_dir, "00_process_videos.py",
                  "00 Process Videos", [str(session)])

        # Step 01: ARCore poses (sensor_data.jsonl → camera_trajectory.csv)
        _run_step(script_dir, "03_convert_arcore_pose.py",
                  "01 Convert ARCore Poses", [str(session)])

        # Step 02: Detect ArUco
        _run_step(script_dir, "04_detect_aruco.py",
                  "02 Detect ArUco",
                  ['--input_dir', str(session / 'demos'),
                   '--camera_intrinsics', str(intrinsics_path),
                   '--aruco_yaml', str(aruco_config_path)])

        # Step 03: Per-demo calibration (+ optional PGO)
        calib_args = ['--ref', ref, str(session)]
        if ignore_gripper_tags:
            calib_args.append('--ignore_gripper_tags')
        if pgo:
            calib_args += [
                '--pgo',
                '--sigma_prior',     str(sigma_prior),
                '--sigma_odom',      str(sigma_odom),
                '--max_center_dist', str(max_center_dist),
                '--min_tag_dist',    str(min_tag_dist),
            ]
        _run_step(script_dir, "05_run_calibrations_per_episode.py",
                  "03 Run Calibrations (ARCore)", calib_args)

        # Step 04: Generate dataset plan
        dataset_plan_args = ['--input', str(session)]
        if ignore_gripper_tags:
            dataset_plan_args.append('--ignore_gripper_tags')
        if pgo:
            dataset_plan_args.append('--use_pgo')
        _run_step(script_dir, "06_generate_dataset_plan.py",
                  "04 Generate Dataset Plan (ARCore)",
                  dataset_plan_args)

        print()
        print("=" * 60)
        print(f"  Pipeline complete for: {session.name}")
        print(f"  Output: {session / 'dataset_plan.pkl'}")
        print("=" * 60)


def _run_step(script_dir, script_name, step_label, args):
    """Run a pipeline step with error checking."""
    print(f"\n--- {step_label} ---")
    script_path = script_dir / script_name
    assert script_path.is_file(), f"Script not found: {script_path}"
    cmd = ['python', str(script_path)] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {step_label}")
        exit(1)


if __name__ == "__main__":
    cli()
