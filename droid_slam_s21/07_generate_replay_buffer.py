"""
Galaxy S21 version of 07_generate_replay_buffer.py

Changes from GoPro version:
- No fisheye converter (S21 has pinhole lens)
- Uses s21_intrinsics_1080p.json instead of gopro_intrinsics_2_7k.json
- mirror masking disabled by default (no mirror on S21 setup)
- Gripper mask still applied (gripper is still in view)

Usage:
    python droid_slam_s21/07_generate_replay_buffer.py \
        <session_dir> -o output.zarr.zip
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from umi.common.cv_util import (
    get_image_transform,
    draw_s21_training_mask,
    inpaint_tag,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg
register_codecs()


# %%
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-cl', '--compression_level', type=int, default=90)
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-mm', '--max_motion', type=float, default=None,
              help='Filter episodes with TCP motion (start→end) > this value in cm. '
                   'Use to drop DROID-SLAM drifted episodes.')
def main(input, output, out_res, compression_level,
         num_workers, max_motion):
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass

    out_res = tuple(int(x) for x in out_res.split(','))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)

    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())

    # dump lowdim data to replay buffer
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue

        plan = pickle.load(plan_path.open('rb'))

        if max_motion is not None:
            n_before = len(plan)
            def _tcp_motion_cm(ep):
                tcp = np.array(ep['grippers'][0]['tcp_pose'])
                return np.linalg.norm(tcp[-1, :3] - tcp[0, :3]) * 100
            plan = [ep for ep in plan if _tcp_motion_cm(ep) <= max_motion]
            print(f"Motion filter ≤{max_motion}cm: kept {len(plan)}/{n_before} episodes")

        videos_dict = defaultdict(list)
        for plan_episode in plan:
            grippers = plan_episode['grippers']

            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)

            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)

            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):
                eef_pose = gripper['tcp_pose']
                eef_pos = eef_pose[...,:3]
                eef_rot = eef_pose[...,3:]
                gripper_widths = gripper['gripper_width']
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']

                robot_name = f'robot{gripper_id}'
                episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
                episode_data[robot_name + '_demo_end_pose'] = demo_end_pose

            out_replay_buffer.add_episode(data=episode_data, compressors=None)

            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()

                video_start, video_end = camera['video_start_end']
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)

                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })
            buffer_start += n_frames

        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())

    print(f"{len(all_videos)} videos used in total!")

    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width

    # dump images
    img_compressor = Jpeg(level=compression_level)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    def video_to_zarr(replay_buffer, mp4_path, tasks):
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]

        curr_task_idx = 0

        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    break

                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']

                    img = frame.to_ndarray(format='rgb24')

                    # inpaint tags
                    this_det = tag_detection_results[frame_idx]
                    all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                    for corners in all_corners:
                        img = inpaint_tag(img, corners)

                    # mask out gripper (S21 training mask)
                    img = draw_s21_training_mask(img, color=(0, 0, 0))

                    # resize (no fisheye conversion for S21)
                    img = resize_tf(img)

                    img_array[buffer_idx] = img
                    buffer_idx += 1

                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        curr_task_idx += 1
                else:
                    assert False

    with tqdm(total=len(vid_args)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(futures,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr,
                    out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print("Done generating replay buffer.")

    # dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")

# %%
if __name__ == "__main__":
    main()
