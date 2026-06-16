# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import numpy as np
import pickle
import json
import pandas as pd
from scipy.spatial.transform import Rotation
from umi.common.pose_util import pose_to_mat
from skfda.exploratory.stats import geometric_median

# %%
@click.command()
@click.option('-d', '--tag_detection', required=True, help='Tag detection pkl path')
@click.option('-c', '--csv_trajectory', default=None, help='CSV trajectory from SLAM (not mapping)')
@click.option('-o', '--output', required=True, help='output json')
@click.option('-tid', '--tag_id', type=int, default=13)
@click.option('-k', '--keyframe_only', is_flag=True, default=False)
@click.option('-iw', '--image_width', type=int, default=2704, help='Image width (pixels)')
@click.option('-ih', '--image_height', type=int, default=2028, help='Image height (pixels)')
@click.option('--max_center_dist', type=float, default=0.9,
              help='Max normalized distance from image center for tag detection filter '
                   '(normalized by half image height). Default 0.9.')
@click.option('--min_tag_dist', type=float, default=0.3,
              help='Minimum distance (m) from camera to tag. Lower for close-up setups (e.g. 0.1 for S21 1x). Default 0.3.')
def main(tag_detection, csv_trajectory, output, tag_id, keyframe_only, image_width, image_height, max_center_dist, min_tag_dist):
    """
    Please use camera_trajectory.csv produced by re-localizing (initializing)
    the mapping video with the map_atlas.osa produced by mapping run.
    This is much more accurate than the mapping_camera_trajectory.csv produced by
    mapping run itself.
    """

    # load
    df = pd.read_csv(csv_trajectory)
    tag_detection_results = pickle.load(open(tag_detection, 'rb'))

    # filter pose
    is_valid = df['is_lost'].astype(str).str.lower() == 'false'
    if keyframe_only:
        is_valid &= df['is_keyframe']

    # convert to mat
    cam_pose_timestamps = df['timestamp'].loc[is_valid].to_numpy()
    cam_pos = df[['x','y','z']].loc[is_valid].to_numpy()
    cam_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].loc[is_valid].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:,3,3] = 1
    cam_pose[:,:3,3] = cam_pos
    cam_pose[:,:3,:3] = cam_rot.as_matrix()

    # match tum data to video idx
    video_timestamps = np.array([x['time'] for x in tag_detection_results])
    tum_video_idxs = list()
    for t in cam_pose_timestamps:
        tum_video_idxs.append(np.argmin(np.abs(video_timestamps - t)))

    # find corresponding tag detection
    all_tx_slam_tag = list()
    all_tx_cam_tag = list()      # tx_cam_tag per frame (metric)
    all_cam_pos_droid = list()   # camera position in SLAM/DROID units
    all_rotated_tag = list()     # R_slam_cam @ t_cam_tag (metric, SLAM world orientation)
    all_idxs = list()
    for tum_idx, video_idx in enumerate(tum_video_idxs):
        td = tag_detection_results[video_idx]
        tag_dict = td['tag_dict']
        if tag_id not in tag_dict:
            continue

        tag = tag_dict[tag_id]
        pose = np.concatenate([tag['tvec'], tag['rvec']])
        tx_cam_tag = pose_to_mat(pose)
        tx_slam_cam = cam_pose[tum_idx]

        # filter cam pose
        dist_to_cam = np.linalg.norm(tx_cam_tag[:3,3])
        if (dist_to_cam < min_tag_dist) or (dist_to_cam > 2):
            continue

        # filter tag location in image
        corners = tag['corners']
        tag_center_pix = corners.mean(axis=0)
        img_center = np.array([image_width, image_height], dtype=np.float32) / 2
        dist_to_center = np.linalg.norm(tag_center_pix - img_center) / img_center[1]
        if dist_to_center > max_center_dist:
            continue

        tx_slam_tag = tx_slam_cam @ tx_cam_tag
        all_tx_slam_tag.append(tx_slam_tag)
        all_tx_cam_tag.append(tx_cam_tag.copy())
        all_cam_pos_droid.append(tx_slam_cam[:3, 3].copy())
        all_rotated_tag.append((tx_slam_cam[:3, :3] @ tx_cam_tag[:3, 3]).copy())
        all_idxs.append(tum_idx)
    all_tx_slam_tag = np.array(all_tx_slam_tag)
    all_tx_cam_tag = np.array(all_tx_cam_tag)
    all_cam_pos_droid = np.array(all_cam_pos_droid)
    all_rotated_tag = np.array(all_rotated_tag)

    # find transform closest to the mean
    all_slam_tag_pos = all_tx_slam_tag[:,:3,3]
    median = geometric_median(all_slam_tag_pos)
    dists = np.linalg.norm((all_tx_slam_tag[:,:3,3] - median), axis=-1)
    threshold = np.quantile(dists, 0.9)
    is_valid = dists < threshold
    std = all_slam_tag_pos[is_valid].std(axis=0)
    mean = all_slam_tag_pos[is_valid].mean(axis=0)
    print("Tag detection standard deviation (cm) < 0.9 quantile")
    print(std * 100)

    # ── Rotation averaging (geodesic mean on SO(3)) ───────────────────────
    # 단일 프레임 rotation 대신 유효 프레임 전체의 rotation을 평균화.
    # scipy Rotation.mean()이 geodesic mean을 계산함.
    valid_rots = Rotation.from_matrix(all_tx_slam_tag[is_valid][:, :3, :3])
    mean_rot = valid_rots.mean()

    # translation: 평균 rotation + 평균 tag position
    tx_slam_tag = np.eye(4, dtype=np.float64)
    tx_slam_tag[:3, :3] = mean_rot.as_matrix()
    tx_slam_tag[:3,  3] = mean

    print(f"Averaged rotation over {is_valid.sum()} frames "
          f"(vs single-frame before). Rotation std: "
          f"{np.degrees(valid_rots.magnitude().std()):.2f} deg")

    # ── Compute DROID-to-metric scale factor ────────────────────────────
    # The ArUco tag is a fixed point in the world. For each frame i:
    #   tag_world_droid = R_i @ (scale * t_cam_tag_i) + cam_pos_droid_i
    # For two frames 0 and k:
    #   scale * (R_0 @ t_0 - R_k @ t_k) = cam_pos_k - cam_pos_0
    # i.e.  scale * a_k = b_k  →  least-squares: scale = Σ a·b / Σ a·a
    cam_pos_valid = all_cam_pos_droid[is_valid]    # (N, 3) DROID units
    rotated_tag_valid = all_rotated_tag[is_valid]  # (N, 3) metric
    scale_factor = 1.0
    if len(cam_pos_valid) >= 2:
        ref_cam_pos = cam_pos_valid[0]
        ref_rotated_tag = rotated_tag_valid[0]
        a = ref_rotated_tag - rotated_tag_valid[1:]   # (N-1, 3) metric
        b = cam_pos_valid[1:] - ref_cam_pos           # (N-1, 3) DROID
        a_norm = np.linalg.norm(a, axis=1)
        enough_motion = a_norm > 1e-4
        if enough_motion.sum() >= 3:
            a_v, b_v = a[enough_motion], b[enough_motion]
            s = float(np.sum(a_v * b_v) / np.sum(a_v * a_v))
            if 0.1 < s < 10.0:
                scale_factor = s
            else:
                print(f"Warning: unreliable scale estimate ({s:.3f}), using 1.0")
        else:
            print("Warning: too few valid frame pairs for scale estimation, using 1.0")
    else:
        print("Warning: too few tag detections for scale estimation, using 1.0")
    print(f"DROID/metric scale factor: {scale_factor:.4f}  "
          f"(metric = DROID / {scale_factor:.4f})")

    # tx_slam_tag의 translation을 metric scale로 보정
    # (rotation 평균은 이미 위에서 계산됨, translation만 scale 적용)
    tx_slam_tag[:3, 3] = mean / scale_factor

    # save
    result = {
        'tx_slam_tag': tx_slam_tag.tolist(),
        'scale_factor': scale_factor,
    }
    json.dump(result, open(output, 'w'), indent=2)
    print(f"Saved result to {output}")


# %%
if __name__ == "__main__":
    main()
