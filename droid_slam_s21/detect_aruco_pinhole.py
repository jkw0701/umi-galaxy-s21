"""
ArUco tag detection for pinhole cameras (Galaxy S21).

The original scripts/detect_aruco.py uses cv2.fisheye.undistortPoints,
which only works for fisheye cameras (GoPro). This version uses
standard pinhole undistortion for S21.

Usage:
    python droid_slam_s21/detect_aruco_pinhole.py \
        --input <video.mp4> \
        --output <tag_detection.pkl> \
        --intrinsics_json <s21_intrinsics_1080p.json> \
        --aruco_yaml <aruco_config.yaml>
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
from tqdm import tqdm
import yaml
import json
import av
import numpy as np
import cv2
import pickle

from umi.common.cv_util import parse_aruco_config


def parse_pinhole_intrinsics(json_data: dict):
    """Parse S21 pinhole intrinsics JSON into K and dist_coeffs."""
    if 'camera_matrix' in json_data and 'dist_coeffs' in json_data:
        K = np.array(json_data['camera_matrix'], dtype=np.float64)
        D = np.array(json_data['dist_coeffs'], dtype=np.float64).reshape(-1)
    else:
        intr = json_data['intrinsics']
        fx = intr['focal_length_x']
        fy = intr['focal_length_y']
        cx = intr['principal_pt_x']
        cy = intr['principal_pt_y']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array(intr.get('dist_coeffs', [0,0,0,0,0]), dtype=np.float64)

    img_w = json_data['image_width']
    img_h = json_data['image_height']
    return K, D, (img_w, img_h)


def scale_intrinsics(K, orig_size, target_size):
    """Scale camera matrix to a different resolution."""
    ow, oh = orig_size
    tw, th = target_size
    sx = tw / ow
    sy = th / oh
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx  # fx
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[1, 2] *= sy  # cy
    return K_scaled


def detect_aruco_pinhole(img, aruco_dict, marker_size_map, K, dist_coeffs):
    """Detect and localize ArUco tags using pinhole camera model."""
    param = cv2.aruco.DetectorParameters()
    param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    corners, ids, _ = cv2.aruco.detectMarkers(
        image=img, dictionary=aruco_dict, parameters=param)

    if len(corners) == 0:
        return dict()

    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue

        marker_size_m = marker_size_map[this_id]

        # Undistort corners using pinhole model
        # cv2.undistortPoints returns (4,1,2) from input (1,4,2)
        undistorted = cv2.undistortPoints(this_corners, K, dist_coeffs, P=K)
        # Reshape back to (1,4,2) for estimatePoseSingleMarkers
        undistorted = undistorted.reshape(1, 4, 2)

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            undistorted, marker_size_m, K, np.zeros((1, 5)))

        tag_dict[this_id] = {
            'rvec': rvec[0].flatten(),
            'tvec': tvec[0].flatten(),
            'corners': this_corners[0]
        }

    return tag_dict


@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-ij', '--intrinsics_json', required=True)
@click.option('-ay', '--aruco_yaml', required=True)
@click.option('-n', '--num_workers', type=int, default=4)
def main(input, output, intrinsics_json, aruco_yaml, num_workers):
    cv2.setNumThreads(num_workers)

    # load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']

    # load pinhole intrinsics
    K, dist_coeffs, orig_size = parse_pinhole_intrinsics(
        json.load(open(intrinsics_json, 'r')))

    results = list()
    with av.open(os.path.expanduser(input)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        in_res = (in_stream.width, in_stream.height)

        # Scale intrinsics if video resolution differs from calibration
        if in_res != orig_size:
            K_scaled = scale_intrinsics(K, orig_size, in_res)
        else:
            K_scaled = K

        for i, frame in tqdm(enumerate(in_container.decode(in_stream)), total=in_stream.frames):
            img = frame.to_ndarray(format='bgr24')
            frame_cts_sec = frame.pts * in_stream.time_base

            tag_dict = detect_aruco_pinhole(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                K=K_scaled,
                dist_coeffs=dist_coeffs
            )
            result = {
                'frame_idx': i,
                'time': float(frame_cts_sec),
                'tag_dict': tag_dict
            }
            results.append(result)

    pickle.dump(results, open(os.path.expanduser(output), 'wb'))
    print(f"Done! Processed {len(results)} frames.")


if __name__ == "__main__":
    main()
