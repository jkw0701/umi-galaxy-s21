"""
DROID-SLAM runner script.
Runs inside the 'droid' conda environment.
Called by 03_batch_slam.py via subprocess.

Output: camera_trajectory.csv with columns:
    frame_idx, timestamp, state, is_lost, is_keyframe, x, y, z, q_x, q_y, q_z, q_w

# 0514 droid slam 파라미터 수정: line 159~168
"""

import sys
import os
import argparse
import pathlib
import numpy as np

# Add DROID-SLAM to path
DROID_SLAM_DIR = pathlib.Path(__file__).parent.parent.parent / 'DROID-SLAM'
sys.path.insert(0, str(DROID_SLAM_DIR))
sys.path.insert(0, str(DROID_SLAM_DIR / 'droid_slam'))

import torch
from tqdm import tqdm
import cv2
from droid import Droid


def image_stream(imagedir, calib, stride=1):
    """Image generator compatible with DROID-SLAM demo.py"""
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    image_list = sorted(pathlib.Path(imagedir).glob('*.png'))[::stride]
    image_list = [str(p) for p in image_list]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(imfile)
        if image is None:
            continue

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def poses_to_csv(traj_est, timestamps, stride, total_frames, output_csv):
    """
    Convert DROID-SLAM traj_est to camera_trajectory.csv

    traj_est: SE3 inverse data, shape (N_keyframes_or_all, 7) [tx, ty, tz, qx, qy, qz, qw]
    timestamps: full frame timestamps (length = total_frames)
    stride: frame stride used in DROID-SLAM
    total_frames: total number of video frames

    DROID-SLAM with stream passed to terminate() returns poses for ALL frames (via PoseTrajectoryFiller).
    """
    import pandas as pd

    n_poses = len(traj_est)

    # traj_est[i] corresponds to frame index i*stride
    # But when terminate(stream) is called, PoseTrajectoryFiller fills ALL frames
    # So n_poses should equal total_frames // stride (approximately)

    rows = []
    for i in range(n_poses):
        frame_idx = i * stride
        if frame_idx >= total_frames:
            break

        ts = timestamps[frame_idx] if frame_idx < len(timestamps) else frame_idx / 30.0
        tx, ty, tz = traj_est[i, :3]
        qx, qy, qz, qw = traj_est[i, 3], traj_est[i, 4], traj_est[i, 5], traj_est[i, 6]

        rows.append({
            'frame_idx': frame_idx,
            'timestamp': ts,
            'state': 2,
            'is_lost': 'false',
            'is_keyframe': 'false',
            'x': tx,
            'y': ty,
            'z': tz,
            'q_x': qx,
            'q_y': qy,
            'q_z': qz,
            'q_w': qw,
        })

    # Fill in intermediate frames (between strides) by linear interpolation
    if stride > 1:
        full_rows = []
        for k in range(len(rows) - 1):
            r0 = rows[k]
            r1 = rows[k + 1]
            full_rows.append(r0)
            # interpolate between r0.frame_idx and r1.frame_idx
            for j in range(1, stride):
                alpha = j / stride
                fidx = r0['frame_idx'] + j
                if fidx >= total_frames:
                    break
                ts = timestamps[fidx] if fidx < len(timestamps) else fidx / 30.0
                full_rows.append({
                    'frame_idx': fidx,
                    'timestamp': ts,
                    'state': 2,
                    'is_lost': 'false',
                    'is_keyframe': 'false',
                    'x': (1 - alpha) * r0['x'] + alpha * r1['x'],
                    'y': (1 - alpha) * r0['y'] + alpha * r1['y'],
                    'z': (1 - alpha) * r0['z'] + alpha * r1['z'],
                    'q_x': (1 - alpha) * r0['q_x'] + alpha * r1['q_x'],
                    'q_y': (1 - alpha) * r0['q_y'] + alpha * r1['q_y'],
                    'q_z': (1 - alpha) * r0['q_z'] + alpha * r1['q_z'],
                    'q_w': (1 - alpha) * r0['q_w'] + alpha * r1['q_w'],
                })
        if rows:
            full_rows.append(rows[-1])
        rows = full_rows

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows -> {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagedir', required=True)
    parser.add_argument('--calib', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--timestamps_npy', required=True)
    parser.add_argument('--weights', default='droid.pth')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--disable_vis', action='store_true')
    parser.add_argument('--buffer', type=int, default=512)
    parser.add_argument('--beta', type=float, default=0.3)

# 0514 하이퍼파라미터 변경 전
#    parser.add_argument('--filter_thresh', type=float, default=2.4)
# 0514 하이퍼파라미터 변경 후 (1줄)
#    parser.add_argument('--filter_thresh', type=float, default=3.4)
# 0520 최적 파라미터
    parser.add_argument('--filter_thresh', type=float, default=2.4)

# 0520 최적 파라미터
    parser.add_argument('--warmup', type=int, default=26)
# 0514 하이퍼파라미터 변경 전
#    parser.add_argument('--keyframe_thresh', type=float, default=4.0)
# 0514 하이퍼파라미터 변경 후 (1줄)
#    parser.add_argument('--keyframe_thresh', type=float, default=1.0)
# 0520 최적 파라미터
    parser.add_argument('--keyframe_thresh', type=float, default=2.8)


    parser.add_argument('--frontend_thresh', type=float, default=16.0)
    parser.add_argument('--frontend_window', type=int, default=25)
    parser.add_argument('--frontend_radius', type=int, default=2)
    parser.add_argument('--frontend_nms', type=int, default=1)
    parser.add_argument('--backend_thresh', type=float, default=22.0)
    parser.add_argument('--backend_radius', type=int, default=2)
    parser.add_argument('--backend_nms', type=int, default=3)
    parser.add_argument('--upsample', action='store_true')
    parser.add_argument('--image_size', default=[240, 320])
    parser.add_argument('--stereo', action='store_true', default=False)
    parser.add_argument('--frontend_device', type=str, default='cuda')
    parser.add_argument('--backend_device', type=str, default='cuda')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn', force=True)

    # Load timestamps
    timestamps = np.load(args.timestamps_npy).tolist()
    total_frames = len(timestamps)

    # Count image files
    image_files = sorted(pathlib.Path(args.imagedir).glob('*.png'))
    print(f"Total frames: {total_frames}, image files: {len(image_files)}")

    # Run DROID-SLAM
    droid = None
    stream = image_stream(args.imagedir, args.calib, args.stride)

    for (t, image, intrinsics) in tqdm(stream, desc='DROID-SLAM tracking'):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        droid.track(t, image, intrinsics=intrinsics)

    if droid is None:
        print("ERROR: No frames processed!")
        sys.exit(1)

    # Terminate and get full trajectory
    print("Running global BA and filling trajectory...")
    traj_est = droid.terminate(
        image_stream(args.imagedir, args.calib, args.stride)
    )
    # traj_est shape: (N, 7) [tx, ty, tz, qx, qy, qz, qw]
    print(f"traj_est shape: {traj_est.shape}")

    # Save as camera_trajectory.csv
    poses_to_csv(
        traj_est=traj_est,
        timestamps=timestamps,
        stride=args.stride,
        total_frames=total_frames,
        output_csv=args.output_csv,
    )


if __name__ == '__main__':
    main()
