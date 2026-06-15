"""
Galaxy S21 calibration setup — run ONCE before data collection.

This script:
1. Takes a checkerboard calibration video from S21
2. Extracts frames and runs OpenCV camera calibration
3. Generates s21_intrinsics_1080p.json (for ArUco detection and DROID-SLAM)
4. Generates s21_1080p_setting.yaml (for ORB_SLAM3, optional)
5. Copies aruco_config.yaml to the output calibration directory

Usage:
    python scripts_slam_s21/setup_s21_calibration.py \
        --video <checkerboard_video.mp4> \
        --output_dir example/calibration_s21

    Optional:
        --checker_cols 9          # inner corners per row
        --checker_rows 6          # inner corners per column
        --square_size 0.025       # square size in meters
        --fps 30                  # S21 recording fps
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import json
import click
import cv2
import numpy as np
import shutil
import av
from tqdm import tqdm


def extract_checkerboard_frames(video_path, checker_size, max_frames=100, skip_interval=10):
    """
    Extract frames from video that contain a valid checkerboard.
    Returns obj_points, img_points, img_size.
    """
    obj_points = []
    img_points = []
    img_size = None

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total_frames = stream.frames

        for frame_idx, frame in tqdm(enumerate(container.decode(stream)),
                                      total=total_frames, desc="Scanning for checkerboard"):
            if frame_idx % skip_interval != 0:
                continue

            img = frame.to_ndarray(format='bgr24')
            if img_size is None:
                img_size = (img.shape[1], img.shape[0])  # (w, h)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checker_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objp = np.zeros((checker_size[0] * checker_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:checker_size[0], 0:checker_size[1]].T.reshape(-1, 2)
                obj_points.append(objp)
                img_points.append(corners_refined)

                if len(obj_points) >= max_frames:
                    break

    print(f"Found {len(obj_points)} valid checkerboard frames out of {total_frames} total")
    return obj_points, img_points, img_size


def calibrate_pinhole(obj_points, img_points, img_size):
    """Run OpenCV pinhole camera calibration. Returns K, dist_coeffs, reproj_error."""
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)
    return K, dist_coeffs, ret


def generate_intrinsics_json(K, dist_coeffs, img_size, fps, reproj_error, nr_images):
    """Generate s21_intrinsics_1080p.json (camera intrinsics in UMI-compatible JSON format)."""
    w, h = img_size
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    data = {
        "final_reproj_error": float(reproj_error),
        "fps": float(fps),
        "image_height": h,
        "image_width": w,
        "intrinsic_type": "PINHOLE",
        "intrinsics": {
            "focal_length_x": fx,
            "focal_length_y": fy,
            "principal_pt_x": cx,
            "principal_pt_y": cy,
            "dist_coeffs": [float(x) for x in dist_coeffs.flatten()[:5]]
        },
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "nr_calib_images": nr_images
    }
    return data


def generate_slam_yaml(K, dist_coeffs, img_size, fps, output_path):
    """Generate ORB_SLAM3 setting YAML for S21 pinhole camera (optional, for ORB-SLAM3 use)."""
    w, h = img_size
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    d = dist_coeffs.flatten()
    k1 = float(d[0]) if len(d) > 0 else 0.0
    k2 = float(d[1]) if len(d) > 1 else 0.0
    p1 = float(d[2]) if len(d) > 2 else 0.0
    p2 = float(d[3]) if len(d) > 3 else 0.0
    k3 = float(d[4]) if len(d) > 4 else 0.0

    yaml_content = f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters (S21 1080p pinhole, auto-generated)
#--------------------------------------------------------------------------------------------
Camera.type: "PinHole"

Camera.fx: {fx:.6f}
Camera.fy: {fy:.6f}
Camera.cx: {cx:.6f}
Camera.cy: {cy:.6f}

Camera.k1: {k1:.8f}
Camera.k2: {k2:.8f}
Camera.p1: {p1:.8f}
Camera.p2: {p2:.8f}
Camera.k3: {k3:.8f}

Camera.width: {w}
Camera.height: {h}
Camera.fps: {fps}

Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# IMU Parameters (reasonable defaults for smartphone IMU, tune if needed)
#--------------------------------------------------------------------------------------------
IMU.NoiseGyro: 1.7e-4
IMU.NoiseAcc: 2.0e-3
IMU.GyroWalk: 1.9e-5
IMU.AccWalk: 3.0e-3
IMU.Frequency: 200

Tbc: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]

IMU.InsertKFsWhenLost: 1

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------
ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content)


@click.command()
@click.option('--video', required=True, help='Checkerboard calibration video from S21 (camera_ultrawide.mp4 or full path)')
@click.option('--output_dir', default='example/calibration_s21',
              show_default=True, help='Output calibration directory')
@click.option('--checker_cols', type=int, default=9, help='Inner corners per row')
@click.option('--checker_rows', type=int, default=6, help='Inner corners per column')
@click.option('--square_size', type=float, default=0.025, help='Square size in meters')
@click.option('--fps', type=float, default=30.0, help='S21 recording FPS')
@click.option('--max_frames', type=int, default=100, help='Max checkerboard frames to use')
@click.option('--skip_interval', type=int, default=10, help='Process every N-th frame')
def main(video, output_dir, checker_cols, checker_rows, square_size,
         fps, max_frames, skip_interval):

    # video 경로가 폴더면 camera_ultrawide.mp4 자동 탐색
    video_path = pathlib.Path(os.path.expanduser(video)).absolute()
    if video_path.is_dir():
        for candidate in ['camera_ultrawide.mp4', 'camera.mp4']:
            p = video_path / candidate
            if p.is_file():
                video_path = p
                print(f"Found video: {video_path}")
                break
        else:
            print(f"No video file found in {video_path}. Expected camera_ultrawide.mp4 or camera.mp4.")
            exit(1)

    assert video_path.is_file(), f"Video not found: {video_path}"

    output_dir = pathlib.Path(os.path.expanduser(output_dir)).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    checker_size = (checker_cols, checker_rows)

    print("=" * 60)
    print("  Step 1: Extracting checkerboard corners from video")
    print("=" * 60)
    obj_points, img_points, img_size = extract_checkerboard_frames(
        video_path, checker_size, max_frames=max_frames, skip_interval=skip_interval)

    if len(obj_points) < 10:
        print(f"Only {len(obj_points)} frames found. Need at least 10.")
        print("Tips: try --skip_interval 5, or check checker_cols/checker_rows match your board.")
        exit(1)

    for op in obj_points:
        op *= square_size

    print("=" * 60)
    print("  Step 2: Running camera calibration")
    print("=" * 60)
    K, dist_coeffs, reproj_error = calibrate_pinhole(obj_points, img_points, img_size)
    print(f"  Reprojection error: {reproj_error:.4f}  (good if < 1.0)")
    print(f"  Camera matrix:\n{K}")
    print(f"  Distortion: {dist_coeffs.flatten()}")

    print("=" * 60)
    print("  Step 3: Saving s21_intrinsics_1080p.json")
    print("=" * 60)
    intrinsics_path = output_dir / 's21_intrinsics_1080p.json'
    intrinsics_data = generate_intrinsics_json(
        K, dist_coeffs, img_size, fps, reproj_error, len(obj_points))
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics_data, f, indent=2)
    print(f"  Saved: {intrinsics_path}")

    print("=" * 60)
    print("  Step 4: Generating s21_1080p_setting.yaml (ORB-SLAM3 용)")
    print("=" * 60)
    slam_yaml_path = output_dir / 's21_1080p_setting.yaml'
    generate_slam_yaml(K, dist_coeffs, img_size, fps, slam_yaml_path)
    print(f"  Saved: {slam_yaml_path}")

    print("=" * 60)
    print("  Step 5: Copying aruco_config.yaml")
    print("=" * 60)
    src_aruco = pathlib.Path(ROOT_DIR) / 'example' / 'calibration_s21' / 'aruco_config.yaml'
    dst_aruco = output_dir / 'aruco_config.yaml'
    if not dst_aruco.is_file():
        if src_aruco.is_file():
            shutil.copy2(src_aruco, dst_aruco)
            print(f"  Copied from calibration_s21: {dst_aruco}")
        else:
            print(f"  Warning: aruco_config.yaml not found. Copy it manually to {output_dir}")
    else:
        print(f"  Already exists, skipping: {dst_aruco}")

    print()
    print("=" * 60)
    print("  Calibration complete!")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"    - s21_intrinsics_1080p.json  ← DROID-SLAM, ArUco 검출에 사용")
    print(f"    - s21_1080p_setting.yaml     ← ORB-SLAM3 용 (DROID-SLAM엔 불필요)")
    print(f"    - aruco_config.yaml")
    print()
    print("  Next: collect S21 demo data, then run:")
    print(f"    python run_slam_pipeline_s21.py process-droid \\")
    print(f"        --calibration_dir {output_dir} \\")
    print(f"        <session_dir>")
    print("=" * 60)


if __name__ == "__main__":
    main()
