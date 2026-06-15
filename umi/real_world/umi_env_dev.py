import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from typing import Dict
from scipy.spatial.transform import Rotation as R

# UMI Framework Components
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols
from umi.common.cv_util import draw_s10_training_mask, draw_s21_training_mask, draw_predefined_mask
from diffusion_policy.common.timestamp_accumulator import TimestampActionAccumulator, ObsAccumulator
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder

# Import the specific controllers we are using
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.DynamixelController import DynamixelController

class UmiEnv:
    def __init__(self,
            # Core parameters from eval script
            output_dir: str,
            robot_config: Dict,
            gripper_config: Dict,
            camera_config: Dict,
            env_config: Dict,
            # Other parameters
            frequency=20,
            obs_image_resolution=(224,224),
            mask_mode: str = 'predefined',
            shm_manager=None
            ):
        # Directory setup
        output_dir = pathlib.Path(output_dir)
        self.video_dir = output_dir.joinpath('videos')
        self.video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # Camera setup
        reset_all_elgato_devices()
        time.sleep(0.1)
        v4l_paths = get_sorted_v4l_paths()
        # Fallback: if no by-id devices found (e.g. Iriun/DroidCam virtual webcam),
        # use /dev/video* directly
        if len(v4l_paths) == 0:
            import glob as _glob
            v4l_paths = sorted(_glob.glob('/dev/video[0-9]*'))
        camera_reorder = camera_config.get('camera_reorder', None)
        if camera_reorder is not None:
            paths = [v4l_paths[i] for i in camera_reorder]
            v4l_paths = paths
        
        camera_resolutions = camera_config.get('resolutions', [[1280,720]] * len(v4l_paths))
        assert len(camera_resolutions) == len(v4l_paths)

        obs_image_resolution = obs_image_resolution  # (W, H)

        def make_camera_transform(input_res):
            tf = get_image_transform(
                input_res=input_res,
                output_res=obs_image_resolution,
                bgr_to_rgb=True)
            def transform(data, _mask_mode=mask_mode):
                img = data['color']
                if _mask_mode == 's10':
                    img = draw_s10_training_mask(img, color=(0, 0, 0))
                elif _mask_mode == 's21':
                    img = draw_s21_training_mask(img, color=(0, 0, 0))
                elif _mask_mode == 'predefined':
                    img = draw_predefined_mask(img, color=(0, 0, 0),
                        mirror=False, gripper=True, finger=False)
                # 'none': no mask
                img = tf(img)
                data['color'] = img
                return data
            return transform

        transforms = [
            make_camera_transform(input_res=(res[0], res[1]))
            for res in camera_resolutions
        ]

        self.camera = MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=[tuple(res) for res in camera_resolutions],
            capture_fps=[60] * len(v4l_paths),
            put_downsample=False,
            get_max_k=env_config.get('max_obs_buffer_size', 60),
            receive_latency=env_config.get('camera_obs_latency', 0.125),
            transform=transforms,
            vis_transform=transforms,
        )
        
        enable_multi_cam_vis = env_config.get('enable_multi_cam_vis', True)
        if enable_multi_cam_vis:
            rw, rh, col, row = optimal_row_cols(
                n_cameras=len(v4l_paths), in_wh_ratio=4/3, 
                max_resolution=env_config.get('multi_cam_vis_resolution', (960, 960)))
            self.multi_cam_vis = MultiCameraVisualizer(
                camera=self.camera, row=row, col=col, rgb_to_bgr=False)
        else:
            self.multi_cam_vis = None

        # Controller Instantiation
        robot_type = robot_config.pop('robot_type', 'franka')
        if robot_type == 'franka':
            #  self.robot = FrankaInterpolationController(
            #     shm_manager=shm_manager,
            #     robot_ip=robot_config['robot_ip'],
            #     receive_latency=robot_config.get('robot_obs_latency', 0.00),
            #     Kx_scale=1.0,
            #     Kxd_scale=1.0,
            #     )
            self.robot = FrankaInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_config['robot_ip'],
                frequency=60,
                Kx_scale=1.5,
                Kxd_scale=np.array([1.5,1.5,1.5,1.0,1.0,1.0]),
                verbose=False,
                receive_latency=robot_config.get('robot_obs_latency', 0.00)
            )
        else:
            raise ValueError(f"Unsupported robot_type: {robot_type}")

        gripper_type = gripper_config.pop('gripper_type', 'dynamixel')
        print(gripper_config['gripper_max_width_mm'], gripper_config['motor_positions_open'], gripper_config['motor_positions_closed'])
        if gripper_type == 'dynamixel':
            self.gripper = DynamixelController(
                shm_manager=shm_manager,
                device_name=gripper_config['device_name'],
                baudrate=gripper_config['baudrate'],
                dxl_ids=gripper_config['dxl_ids'],
                # Map YAML key 'gripper_obs_latency' to constructor argument 'receive_latency'
                # receive_latency=gripper_config.get('gripper_obs_latency', 0.0)
                gripper_max_width_mm=gripper_config['gripper_max_width_mm'],
                motor_positions_open=gripper_config['motor_positions_open'],
                motor_positions_closed=gripper_config['motor_positions_closed']
            )
        else:
            raise ValueError(f"Unsupported gripper_type: {gripper_type}")

        # Store parameters
        self.frequency = frequency
        self.align_camera_idx = env_config.get('align_camera_idx', 0)
        self.camera_obs_horizon = env_config.get('camera_obs_horizon', 2)
        self.robot_obs_horizon = env_config.get('robot_obs_horizon', 2)
        self.gripper_obs_horizon = env_config.get('gripper_obs_horizon', 2)
        self.output_dir = output_dir
        self.last_camera_data = None
        self.obs_accumulator = None
        self.action_accumulator = None
        self.start_time = None
        self.robot_action_latency = robot_config.get('action_latency', 0.1)
        self.gripper_action_latency = gripper_config.get('action_latency', 0.1)
        self.episode_start_pose = None

    @property
    def is_ready(self):
        c_ready = self.camera.is_ready
        r_ready = self.robot.is_ready
        g_ready = self.gripper.is_ready
        
        if not (c_ready and r_ready and g_ready):
            print(f"\n[🚨 비상 🚨] is_ready 실패! 상태 보고 ➜ 카메라: {c_ready} | 로봇: {r_ready} | 그리퍼: {g_ready}\n")
            
        return c_ready and r_ready and g_ready
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        self.gripper.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        self.gripper.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.gripper.stop_wait()
        self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.stop()

    def get_obs(self) -> dict:
        assert self.is_ready

        k = math.ceil(self.camera_obs_horizon * (60 / self.frequency)) + 2
        self.last_camera_data = self.camera.get(k=k, out=self.last_camera_data)
        last_robot_data = self.robot.get_all_state()
        last_gripper_data = self.gripper.get_all_state()

        last_timestamp = self.last_camera_data[self.align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        camera_obs_timestamps = last_timestamp - (np.arange(self.camera_obs_horizon)[::-1] * dt)
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            cam_timestamps = value['timestamp']
            cam_idxs = [np.argmin(np.abs(cam_timestamps - t)) for t in camera_obs_timestamps]
            camera_obs[f'camera{camera_idx}_rgb'] = value['color'][cam_idxs]

        robot_obs_timestamps = last_timestamp - (np.arange(self.robot_obs_horizon)[::-1] * dt)
        robot_pose_interpolator = PoseInterpolator(
            t=last_robot_data['robot_timestamp'], x=last_robot_data['ActualTCPPose'])
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose[...,3:]
        }

        # This logic is working in get_umi_obs_dict function
        # if self.episode_start_pose is not None:
        #     start_rot = R.from_rotvec(self.episode_start_pose[3:])
        #     current_rots = R.from_rotvec(robot_obs['robot0_eef_rot_axis_angle'])
        #     rel_rots = current_rots * start_rot.inv()
        #     robot_obs['robot0_eef_rot_axis_angle_wrt_start'] = rel_rots.as_rotvec()
        # else:
        #     robot_obs['robot0_eef_rot_axis_angle_wrt_start'] = np.zeros_like(robot_obs['robot0_eef_rot_axis_angle'])

        gripper_obs_timestamps = last_timestamp - (np.arange(self.gripper_obs_horizon)[::-1] * dt)
        gripper_interpolator = get_interp1d(
            t=last_gripper_data['gripper_timestamp'],
            x=last_gripper_data['gripper_position_mm'][...,None]
        )
        gripper_obs = {
            'robot0_gripper_width': gripper_interpolator(gripper_obs_timestamps)
        }

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': last_robot_data['ActualTCPPose'],
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )
            self.obs_accumulator.put(
                data={
                    'robot0_gripper_width': last_gripper_data['gripper_position_mm'][...,None]
                },
                timestamps=last_gripper_data['gripper_timestamp']
            )

        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data['timestamp'] = camera_obs_timestamps
        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        
        if not isinstance(actions, np.ndarray): actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray): timestamps = np.array(timestamps)

        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        if len(new_actions) == 0:
            return
            
        r_latency = self.robot_action_latency if compensate_latency else 0.0
        g_latency = self.gripper_action_latency if compensate_latency else 0.0

        r_actions = new_actions[:, :6]
        g_actions = new_actions[:, 6]

        for i in range(len(new_actions)):
            self.robot.schedule_waypoint(
                pose=r_actions[i],
                target_time=new_timestamps[i] - r_latency
            )
            self.gripper.schedule_waypoint(
                pos=g_actions[i],
                target_time=new_timestamps[i] - g_latency
            )

        if self.action_accumulator is not None:
            self.action_accumulator.put(new_actions, new_timestamps)
    
    def get_robot_state(self):
        return self.robot.get_state()

    def get_gripper_state(self):
        return self.gripper.get_state()

    def start_episode(self, start_time=None):
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time
        assert self.is_ready

        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)

        n_cameras = self.camera.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1/self.frequency)
        print(f'Episode {episode_id} started!')

        time.sleep(0.1) 
        robot_state = self.robot.get_state()
        self.episode_start_pose = robot_state['ActualTCPPose'][-1].copy()
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.camera.stop_recording()
        
        if self.obs_accumulator is not None:
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                if len(value) > 0:
                    end_time = min(end_time, value[-1])
            if len(self.action_accumulator.timestamps) > 0:
                end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1
            
            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)
                episode['robot0_eef_pos'] = robot_pose[:,:3]
                episode['robot0_eef_rot_axis_angle'] = robot_pose[:,3:]
                joint_pos_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
                    np.array(self.obs_accumulator.data['robot0_joint_pos'])
                )
                joint_vel_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
                    np.array(self.obs_accumulator.data['robot0_joint_vel'])
                )
                episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
                episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)

                gripper_interpolator = get_interp1d(
                    t=np.array(self.obs_accumulator.timestamps['robot0_gripper_width']),
                    x=np.array(self.obs_accumulator.data['robot0_gripper_width'])
                )
                episode['robot0_gripper_width'] = gripper_interpolator(timestamps)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
