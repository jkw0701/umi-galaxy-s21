"""
DROID-SLAM version of 03_batch_slam.py for Galaxy S21.

Changes from ORB-SLAM3 version:
- No Docker, no map file needed
- Uses DROID-SLAM (deep learning based) instead of ORB-SLAM3
- Input: raw_video.mp4 → extracted frames → DROID-SLAM → camera_trajectory.csv
- Runs in a separate conda environment (droid_slam)
- No IMU required

Output camera_trajectory.csv columns:
    frame_idx, timestamp, state, is_lost, is_keyframe, x, y, z, q_x, q_y, q_z, q_w

Usage:
    python droid_slam_s21/03_batch_slam.py -i <demos_dir> --intrinsics <s21_intrinsics.json>
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
import json
import shutil
import tempfile
import numpy as np
import av
import cv2
from tqdm import tqdm

import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from umi.common.cv_util import draw_s21_slam_mask


DROID_SLAM_DIR = pathlib.Path.home() / 'DROID-SLAM'
DROID_CONDA_ENV = 'droid'


def extract_frames(video_path: pathlib.Path, output_dir: pathlib.Path,
                   fps: float = None, use_mask: bool = True):
    """Extract frames from mp4 to output_dir as PNG files.

    use_mask: if True, black out the gripper/finger region before saving.
    This prevents DROID-SLAM from tracking gripper motion as camera motion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_timestamps = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        video_fps = float(stream.average_rate)
        if fps is None:
            fps = video_fps

        for frame_idx, frame in enumerate(container.decode(stream)):
            img = frame.to_ndarray(format='rgb24')
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if use_mask:
                img_bgr = draw_s21_slam_mask(img_bgr, color=(0, 0, 0))

            out_path = output_dir / f'{frame_idx:06d}.png'
            cv2.imwrite(str(out_path), img_bgr)
            frame_timestamps.append(frame_idx / fps)

    return frame_timestamps, fps


def make_calib_file(intrinsics_json: pathlib.Path, output_path: pathlib.Path):
    """Convert s21_intrinsics_1080p.json to DROID-SLAM calib txt format.
    Format: fx fy cx cy [k1 k2 p1 p2 [k3]]
    """
    with open(intrinsics_json, 'r') as f:
        data = json.load(f)

    intr = data['intrinsics']
    fx = intr['focal_length_x']
    fy = intr['focal_length_y']
    cx = intr['principal_pt_x']
    cy = intr['principal_pt_y']
    dist = intr.get('dist_coeffs', [])

    line = f"{fx} {fy} {cx} {cy}"
    if len(dist) >= 4:
        line += f" {dist[0]} {dist[1]} {dist[2]} {dist[3]}"
        if len(dist) >= 5:
            line += f" {dist[4]}"

    output_path.write_text(line)


def run_droid_slam(frames_dir: pathlib.Path, calib_path: pathlib.Path,
                   output_csv: pathlib.Path, frame_timestamps: list,
                   weights_path: pathlib.Path, stride: int = 1,
                   stdout_path: pathlib.Path = None,
                   stderr_path: pathlib.Path = None,
                   buffer: int = 512,
                   frontend_window: int = 25):
    """Run DROID-SLAM via a helper script in the droid conda env."""

    runner_path = pathlib.Path(__file__).parent / '_droid_runner.py'

    cmd = [
        'conda', 'run', '-n', DROID_CONDA_ENV, '--no-capture-output',
        'python', str(runner_path),
        '--imagedir', str(frames_dir),
        '--calib', str(calib_path),
        '--output_csv', str(output_csv),
        '--timestamps_npy', str(frames_dir / 'timestamps.npy'),
        '--weights', str(weights_path),
        '--stride', str(stride),
        '--disable_vis',
        '--buffer', str(buffer),
        '--frontend_window', str(frontend_window),
    ]

    np.save(str(frames_dir / 'timestamps.npy'), np.array(frame_timestamps))

    stdout_f = stdout_path.open('w') if stdout_path else None
    stderr_f = stderr_path.open('w') if stderr_path else None

    try:
        result = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f, timeout=600)
    finally:
        if stdout_f:
            stdout_f.close()
        if stderr_f:
            stderr_f.close()

    return result


def process_video(video_dir: pathlib.Path, calib_path: pathlib.Path,
                  weights_path: pathlib.Path, stride: int, keep_frames: bool,
                  use_mask: bool = True, buffer: int = 512, frontend_window: int = 25):
    """Full pipeline for one video directory."""
    video_path = video_dir / 'raw_video.mp4'
    output_csv = video_dir / 'camera_trajectory.csv'

    if output_csv.is_file():
        print(f"  Skipping {video_dir.name}: camera_trajectory.csv already exists")
        return True

    frames_dir = video_dir / '_droid_frames'
    stdout_path = video_dir / 'droid_stdout.txt'
    stderr_path = video_dir / 'droid_stderr.txt'

    try:
        # Step 1: extract frames
        print(f"  [{video_dir.name}] Extracting frames (mask={'on' if use_mask else 'off'})...")
        frame_timestamps, fps = extract_frames(video_path, frames_dir, use_mask=use_mask)
        print(f"  [{video_dir.name}] {len(frame_timestamps)} frames @ {fps:.1f}fps")

        # Step 2: run DROID-SLAM
        print(f"  [{video_dir.name}] Running DROID-SLAM...")
        result = run_droid_slam(
            frames_dir=frames_dir,
            calib_path=calib_path,
            output_csv=output_csv,
            frame_timestamps=frame_timestamps,
            weights_path=weights_path,
            stride=stride,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            buffer=buffer,
            frontend_window=frontend_window,
        )

        if result.returncode != 0:
            print(f"  [{video_dir.name}] DROID-SLAM FAILED (returncode={result.returncode})")
            print(f"    Check: {stderr_path}")
            return False

        if not output_csv.is_file():
            print(f"  [{video_dir.name}] FAILED: camera_trajectory.csv not created")
            return False

        print(f"  [{video_dir.name}] Done -> {output_csv.name}")
        return True

    finally:
        if not keep_frames and frames_dir.exists():
            shutil.rmtree(frames_dir)


@click.command()
@click.option('-i', '--input_dir', required=True, help='demos/ directory')
@click.option('--intrinsics', required=True, help='Path to s21_intrinsics_1080p.json')
@click.option('--weights', default=None, help='Path to droid.pth (default: DROID-SLAM/droid.pth)')
@click.option('-n', '--num_workers', type=int, default=1,
              help='Parallel workers (keep 1 for GPU memory)')
@click.option('--stride', type=int, default=1,
              help='Frame stride for DROID-SLAM (1=all frames)')
@click.option('--keep_frames', is_flag=True, default=False,
              help='Keep extracted frame images after SLAM (for debugging)')
@click.option('--no_mask', is_flag=True, default=False,
              help='Disable gripper masking (not recommended: gripper motion contaminates SLAM).')
@click.option('--buffer', type=int, default=512,
              help='DROID-SLAM frame buffer size. Reduce (e.g. 256) if CUDA OOM occurs.')
@click.option('--frontend_window', type=int, default=25,
              help='DROID-SLAM frontend window size. Reduce (e.g. 16) if CUDA OOM occurs.')
def main(input_dir, intrinsics, weights, num_workers, stride, keep_frames, no_mask, buffer, frontend_window):
    """Run DROID-SLAM on all demo videos to produce camera_trajectory.csv."""
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    intrinsics_path = pathlib.Path(os.path.expanduser(intrinsics)).absolute()

    assert input_dir.is_dir(), f"Not found: {input_dir}"
    assert intrinsics_path.is_file(), f"Not found: {intrinsics_path}"

    if weights is None:
        weights_path = DROID_SLAM_DIR / 'droid.pth'
    else:
        weights_path = pathlib.Path(os.path.expanduser(weights)).absolute()
    assert weights_path.is_file(), f"droid.pth not found: {weights_path}\nDownload from https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh"

    # Make calib file (reuse across all videos)
    calib_path = input_dir / '_s21_calib.txt'
    make_calib_file(intrinsics_path, calib_path)
    print(f"Calib: {calib_path.read_text()}")

    # Find all demo video directories (mapping dir not needed for DROID-SLAM)
    video_dirs = sorted([x.parent for x in input_dir.glob('demo*/raw_video.mp4')])
    print(f"Found {len(video_dirs)} demo video directories")

    use_mask = not no_mask
    if use_mask:
        print("Gripper mask: ON (gripper/finger region will be blacked out for SLAM)")
    else:
        print("Gripper mask: OFF")

    # Process videos
    results = []
    if num_workers == 1:
        for vdir in tqdm(video_dirs):
            ok = process_video(vdir, calib_path, weights_path, stride, keep_frames, use_mask,
                               buffer=buffer, frontend_window=frontend_window)
            results.append(ok)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_video, vdir, calib_path, weights_path, stride, keep_frames, use_mask,
                                buffer, frontend_window): vdir
                for vdir in video_dirs
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                results.append(future.result())

    success = sum(results)
    print(f"\nDone: {success}/{len(video_dirs)} succeeded")
    if success == 0:
        print("ERROR: All SLAM runs failed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
