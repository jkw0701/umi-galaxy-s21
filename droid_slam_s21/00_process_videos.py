"""
Galaxy S21 version of 00_process_videos.py

Supports two input structures:

[Mode A] App pre-organized (new app):
    <session_dir>/
        mapping/
            camera.mp4
            sensor_data.jsonl
            metadata.json
            frame_timestamps.csv
        gripper_calibration/
            camera.mp4  ...
        session_YYYYMMDD_HHMMSS/
            camera.mp4  ...

[Mode B] Flat files (old app / manual placement):
    <session_dir>/
        camera.mp4
        arcore_video.mp4
        sensor_data.jsonl
        metadata.json

Output (both modes):
    <session_dir>/demos/
        mapping/raw_video.mp4  + companion files
        gripper_calibration_<device>_<datetime>/raw_video.mp4
        demo_<device>_<datetime>/raw_video.mp4

Usage:
    python droid_slam_s21/00_process_videos.py <session_dir>
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import shutil
import json
import datetime

COMPANION_FILES = ['metadata.json', 'imu.csv', 'sensor_data.jsonl', 'frame_timestamps.csv']


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def read_metadata(folder: pathlib.Path) -> dict:
    meta_path = folder / 'metadata.json'
    if meta_path.is_file():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def get_device_id(folder: pathlib.Path) -> str:
    meta = read_metadata(folder)
    # new app format: {"device": "Galaxy S21 5G"}
    if 'device' in meta:
        return meta['device'].replace(' ', '_')
    # old app format: {"device_id": "..."}
    if 'device_id' in meta:
        return meta['device_id']
    return 'S21_DEFAULT'


def get_start_datetime(folder: pathlib.Path) -> datetime.datetime:
    meta = read_metadata(folder)
    # new app format: start_timestamp_ns
    if 'start_timestamp_ns' in meta:
        return datetime.datetime.fromtimestamp(meta['start_timestamp_ns'] / 1e9)
    # old app format: start_time_epoch_ms
    if 'start_time_epoch_ms' in meta:
        return datetime.datetime.fromtimestamp(meta['start_time_epoch_ms'] / 1000.0)
    # fallback
    return datetime.datetime.now()


def copy_companions(src_folder: pathlib.Path, dst_folder: pathlib.Path):
    for companion in COMPANION_FILES:
        src = src_folder / companion
        if src.is_file():
            shutil.copy2(src, dst_folder / companion)


def move_video_to_demo(src_mp4: pathlib.Path, dst_dir: pathlib.Path):
    """Move camera.mp4 → dst_dir/raw_video.mp4 and copy companion files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_video = dst_dir / 'raw_video.mp4'
    if dst_video.is_file():
        print(f"  Already exists, skipping: {dst_dir.name}/raw_video.mp4")
        return
    shutil.copy2(src_mp4, dst_video)
    copy_companions(src_mp4.parent, dst_dir)
    print(f"  → {dst_dir.name}/")


# ──────────────────────────────────────────────
# Mode A: App pre-organized structure
# ──────────────────────────────────────────────

VIDEO_CANDIDATES = ['camera_ultrawide.mp4', 'camera.mp4']


def find_video(folder: pathlib.Path):
    """Return the first existing video file among known candidates."""
    for name in VIDEO_CANDIDATES:
        p = folder / name
        if p.is_file():
            return p
    return None


def is_preorganized(session: pathlib.Path) -> bool:
    """Return True if session has app-organized subfolders."""
    for name in ['mapping', 'gripper_calibration']:
        p = session / name
        if p.is_dir() and (p / 'sensor_data.jsonl').is_file():
            return True
    # new app: 20260403_HHMMSS pattern
    for p in session.glob('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*/'):
        if (p / 'sensor_data.jsonl').is_file():
            return True
    # old app: session_YYYYMMDD_HHMMSS pattern
    for p in session.glob('session_*/'):
        if (p / 'sensor_data.jsonl').is_file():
            return True
    return False


def process_preorganized(session: pathlib.Path, output_dir: pathlib.Path):
    """Handle sessions already organized by the app into named subfolders."""
    print("Detected app pre-organized structure.")

    # mapping/
    mapping_src = session / 'mapping'
    if mapping_src.is_dir():
        mp4 = find_video(mapping_src)
        if mp4 is not None:
            move_video_to_demo(mp4, output_dir / 'mapping')
        else:
            print(f"  Warning: no video in mapping/ — skipping (add the video file first)")

    # gripper_calibration/
    gripper_src = session / 'gripper_calibration'
    if gripper_src.is_dir():
        mp4 = find_video(gripper_src)
        if mp4 is not None:
            device_id = get_device_id(gripper_src)
            dt = get_start_datetime(gripper_src)
            out_dname = 'gripper_calibration_' + device_id + '_' + dt.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
            move_video_to_demo(mp4, output_dir / out_dname)
        else:
            print(f"  Warning: no video in gripper_calibration/ — skipping")

    # new app: 20260403_HHMMSS/ → demo
    for session_folder in sorted(session.glob('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*/')):
        if not session_folder.is_dir():
            continue
        mp4 = find_video(session_folder)
        if mp4 is not None:
            device_id = get_device_id(session_folder)
            dt = get_start_datetime(session_folder)
            out_dname = 'demo_' + device_id + '_' + dt.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
            move_video_to_demo(mp4, output_dir / out_dname)
        else:
            print(f"  Warning: no video in {session_folder.name}/ — skipping")

    # old app: session_*/ → demo
    for session_folder in sorted(session.glob('session_*/')):
        if not session_folder.is_dir():
            continue
        mp4 = find_video(session_folder)
        if mp4 is not None:
            device_id = get_device_id(session_folder)
            dt = get_start_datetime(session_folder)
            out_dname = 'demo_' + device_id + '_' + dt.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
            move_video_to_demo(mp4, output_dir / out_dname)
        else:
            print(f"  Warning: no video in {session_folder.name}/ — skipping")


# ──────────────────────────────────────────────
# Mode B: Flat / raw_videos structure (old app)
# ──────────────────────────────────────────────

def get_s21_start_datetime(mp4_path: pathlib.Path) -> datetime.datetime:
    return get_start_datetime(mp4_path.parent)


def get_s21_device_id(mp4_path: pathlib.Path) -> str:
    return get_device_id(mp4_path.parent)


def process_flat(session: pathlib.Path, input_dir: pathlib.Path, output_dir: pathlib.Path):
    """Handle flat / raw_videos-based structure (original logic)."""

    # create raw_videos if doesn't exist
    if not input_dir.is_dir():
        input_dir.mkdir()
        print(f"raw_videos subdir doesn't exist! Creating one and moving all mp4 videos inside.")
        for mp4_path in list(session.glob('*.mp4')) + list(session.glob('*.MP4')):
            vid_stem = mp4_path.stem
            vid_subdir = input_dir / vid_stem
            vid_subdir.mkdir(parents=True, exist_ok=True)
            out_path = vid_subdir / mp4_path.name
            shutil.move(mp4_path, out_path)
            copy_companions(mp4_path.parent, vid_subdir)

    # create mapping video if doesn't exist
    # Prefer camera.mp4 over arcore_video.mp4.
    mapping_vid_path = input_dir / 'mapping.mp4'
    if not mapping_vid_path.exists() and not mapping_vid_path.is_symlink():
        all_cands = list(input_dir.glob('**/*.mp4')) + list(input_dir.glob('**/*.MP4'))
        camera_cands = [p for p in all_cands if p.name == 'camera.mp4']
        if camera_cands:
            max_path = max(camera_cands, key=lambda p: p.stat().st_size)
        else:
            non_arcore = [p for p in all_cands if p.name != 'arcore_video.mp4'] or all_cands
            max_path = max(non_arcore, key=lambda p: p.stat().st_size) if non_arcore else None
        if max_path is not None:
            source_dir = max_path.parent
            shutil.move(max_path, mapping_vid_path)
            print(f"raw_videos/mapping.mp4 doesn't exist! Renaming {max_path.name}.")
            copy_companions(source_dir, input_dir)

    # create gripper calibration directory if doesn't exist
    gripper_cal_dir = input_dir / 'gripper_calibration'
    if not gripper_cal_dir.is_dir():
        gripper_cal_dir.mkdir()
        print("raw_videos/gripper_calibration doesn't exist! Creating one with the first video of each device.")
        device_start_dict = {}
        device_path_dict = {}
        for mp4_path in list(input_dir.glob('**/*.mp4')) + list(input_dir.glob('**/*.MP4')):
            if mp4_path.name.startswith('map'):
                continue
            start_date = get_s21_start_datetime(mp4_path)
            device_id = get_s21_device_id(mp4_path)
            if device_id not in device_start_dict or start_date < device_start_dict[device_id]:
                device_start_dict[device_id] = start_date
                device_path_dict[device_id] = mp4_path

        for device_id, path in device_path_dict.items():
            print(f"Selected {path.name} for device {device_id}")
            source_dir = path.parent
            out_path = gripper_cal_dir / path.name
            shutil.move(path, out_path)
            copy_companions(source_dir, gripper_cal_dir)

    # second pass: move all remaining mp4s to demos/
    all_mp4_paths = list(input_dir.glob('**/*.mp4')) + list(input_dir.glob('**/*.MP4'))
    input_mp4_paths = []
    for mp4_path in all_mp4_paths:
        if mp4_path.name == 'arcore_video.mp4' and (mp4_path.parent / 'camera.mp4').is_file():
            print(f"Skipping {mp4_path.name} (using camera.mp4 instead)")
            continue
        input_mp4_paths.append(mp4_path)
    print(f'Found {len(input_mp4_paths)} MP4 videos')

    for mp4_path in input_mp4_paths:
        if mp4_path.is_symlink():
            continue

        start_date = get_s21_start_datetime(mp4_path)
        device_id = get_s21_device_id(mp4_path)
        out_dname = 'demo_' + device_id + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

        if mp4_path.name.startswith('mapping'):
            out_dname = "mapping"
        elif mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
            out_dname = "gripper_calibration_" + device_id + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

        this_out_dir = output_dir / out_dname
        this_out_dir.mkdir(parents=True, exist_ok=True)

        out_video_path = this_out_dir / 'raw_video.mp4'
        shutil.move(mp4_path, out_video_path)
        copy_companions(mp4_path.parent, this_out_dir)

        # symlink back
        dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
        rel_path = str(out_video_path.relative_to(session))
        mp4_path.symlink_to(os.path.join(dots, rel_path))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

@click.command(help='Session directories containing S21 recordings.')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        output_dir = session / 'demos'
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_preorganized(session):
            process_preorganized(session, output_dir)
        else:
            process_flat(session, session / 'raw_videos', output_dir)


# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
