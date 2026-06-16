import sys
import os
import cv2
import numpy as np
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

np.set_printoptions(precision=4, suppress=True, linewidth=200, edgeitems=10)

# pos 키워드 (m → mm 변환 대상)
POS_KEYS = ('pos', 'gripper_width')
# axis_angle 키워드 (rad → deg 변환 대상)
ROT_KEYS = ('rot_axis_angle',)


def format_val(key, val):
    """키 이름에 따라 단위 변환 후 문자열 반환."""
    if not isinstance(val, np.ndarray):
        return str(val), key

    arr = val.flatten()

    if any(k in key for k in POS_KEYS):
        arr = arr * 1000.0
        label = f"{key}[mm]"
    elif any(k in key for k in ROT_KEYS):
        arr = np.degrees(arr)
        label = f"{key}[deg]"
    else:
        label = key

    val_str = np.array2string(arr, separator=', ', precision=4)
    return val_str, label


def visualize_zarr(zarr_path, scale_factor=2.0, start_ep=0):
    print(f"Loading Zarr from: {zarr_path}")

    buff = ReplayBuffer.create_from_path(zarr_path, mode='r')

    print(f"Total Episodes: {buff.n_episodes}")
    print(f"Total Steps: {buff.n_steps}")
    print("Data Keys available:", list(buff.data.keys()))

    for i in range(start_ep, buff.n_episodes):
        print(f"\n=== Playing Episode {i} ===")
        episode = buff.get_episode(i)

        cam_keys = sorted([k for k in episode.keys() if 'rgb' in k])
        n_frames = episode[cam_keys[0]].shape[0]

        for t in range(n_frames):
            imgs = []
            for cam_k in cam_keys:
                img = episode[cam_k][t]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                new_w = int(img_bgr.shape[1] * scale_factor)
                new_h = int(img_bgr.shape[0] * scale_factor)
                img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                imgs.append(img_resized)

            vis_img = np.hstack(imgs)

            info_lines = []
            info_lines.append(f"Ep:{i} / Frm:{t} / Total:{n_frames}")

            other_keys = sorted([k for k in episode.keys() if 'rgb' not in k])

            print(f"\n--- Frame {t} Data ---")
            for key in other_keys:
                val = episode[key][t]
                val_str, label = format_val(key, val)
                line_text = f"{label}: {val_str}"
                info_lines.append(line_text)
                print(line_text)

            y_start = 30
            y_step = 25

            for idx, line in enumerate(info_lines):
                y_pos = y_start + (idx * y_step)
                cv2.putText(vis_img, line, (15+1, y_pos+1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(vis_img, line, (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 255, 50), 1)

            cv2.imshow('Zarr Visualization (Expanded)', vis_img)

            key = cv2.waitKey(33)
            if key == ord('q'):
                print("Exiting...")
                return
            elif key == ord('n'):
                print("Skipping to next episode...")
                break

    print("End of Data.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python scripts_slam_s21/visualize_zarr.py <dataset.zarr.zip> [start_ep]")
    else:
        start_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        visualize_zarr(sys.argv[1], scale_factor=4.0, start_ep=start_ep)
