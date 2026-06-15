"""
Convert ARCore ar_pose records from sensor_data.jsonl to camera_trajectory.csv.
Galaxy S21 ARCore pipeline version.

sensor_data.jsonl의 ar_pose 레코드:
    {"type": "ar_pose", "ts": ..., "pos": [x,y,z], "quat": [qx,qy,qz,qw],
     "tracking": "TRACKING", "frame": <frame_idx>}

Output camera_trajectory.csv columns:
    timestamp, x, y, z, q_x, q_y, q_z, q_w, is_lost

Usage:
    python droid_slam_s21/03_convert_arcore_pose.py <session_dir> [<session_dir2> ...]
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import json
import concurrent.futures

import click
import av
import pandas as pd


def process_video_dir(video_dir: pathlib.Path):
    out_csv = video_dir / 'camera_trajectory.csv'
    if out_csv.is_file():
        print(f"Skipping {video_dir.name}, camera_trajectory.csv already exists.")
        return

    jsonl_path = video_dir / 'sensor_data.jsonl'
    if not jsonl_path.is_file():
        print(f"Skipping {video_dir.name}, no sensor_data.jsonl.")
        return

    mp4_path = video_dir / 'raw_video.mp4'
    if not mp4_path.is_file():
        print(f"Skipping {video_dir.name}, no raw_video.mp4.")
        return

    pose_by_frame = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('type') == 'ar_pose':
                frame_idx = record.get('frame')
                if frame_idx is not None:
                    pose_by_frame[frame_idx] = record

    if not pose_by_frame:
        print(f"Skipping {video_dir.name}, no ar_pose records in sensor_data.jsonl.")
        return

    with av.open(str(mp4_path), 'r') as container:
        stream = container.streams.video[0]
        n_frames = stream.frames
        fps = float(stream.average_rate)

    rows = []
    for frame_idx in range(n_frames):
        timestamp = frame_idx / fps
        record = pose_by_frame.get(frame_idx)
        if record is not None and record.get('tracking') == 'TRACKING':
            pos = record['pos']
            quat = record['quat']
            x, y, z = pos[0], pos[1], pos[2]
            q_x, q_y, q_z, q_w = quat[0], quat[1], quat[2], quat[3]
            is_lost = 'false'
        else:
            x, y, z = 0.0, 0.0, 0.0
            q_x, q_y, q_z, q_w = 0.0, 0.0, 0.0, 1.0
            is_lost = 'true'
        rows.append({
            'timestamp': timestamp,
            'x': x, 'y': y, 'z': z,
            'q_x': q_x, 'q_y': q_y, 'q_z': q_z, 'q_w': q_w,
            'is_lost': is_lost
        })

    df = pd.DataFrame(rows, columns=['timestamp', 'x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w', 'is_lost'])
    df.to_csv(out_csv, index=False)
    print(f"Wrote camera_trajectory.csv for {video_dir.name} ({n_frames} frames, fps={fps:.3f})")


@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('--num_workers', type=int, default=4, help='Number of parallel workers')
def main(session_dir, num_workers):
    video_dirs = []
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        for mp4_path in sorted(session.glob('demos/*/raw_video.mp4')):
            video_dirs.append(mp4_path.parent)

    if not video_dirs:
        print("No video directories found.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_video_dir, video_dirs))


if __name__ == '__main__':
    main()
