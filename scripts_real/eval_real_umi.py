"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

#from moviepy.editor import VideoFileClip
import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.umi_env_dev import UmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)

from umi.common.pose_util import pose_to_mat, mat_to_pose
# from umi.real_world.spacemouse_shared_memory import Spacemouse

# Define Ensembler
class TemporalEnsembler:
    def __init__(self, max_steps: int, action_dim: int):
        """
        max_steps: 앙상블을 수행할 최대 미래 step 길이 (예: n_action_steps * 2)
        action_dim: 액션의 차원 (예: 7 or 10)
        """
        self.max_steps = max_steps
        self.action_dim = action_dim
        # 누적 합을 저장할 버퍼
        self.action_sum = np.zeros((max_steps, action_dim), dtype=np.float32)
        # 몇 번 더해졌는지 카운트할 버퍼
        self.action_count = np.zeros((max_steps, 1), dtype=np.float32)

    def update(self, new_action_chunk: np.ndarray):
        """
        새로운 추론 결과(action_chunk)를 받아서 버퍼에 누적
        new_action_chunk shape: (T, D)
        """
        length = new_action_chunk.shape[0]
        # 버퍼 범위를 넘어가면 자름
        length = min(length, self.max_steps)
        
        # 현재 버퍼의 해당 위치에 더하기
        self.action_sum[:length] += new_action_chunk[:length]
        self.action_count[:length] += 1.0

    def get_next_actions(self, steps: int) -> np.ndarray:
        """
        실행할 steps만큼의 액션을 꺼내고(평균 계산), 버퍼를 앞으로 당김(Shift)
        """
        # 1. 현재 버퍼에서 평균 계산 (Sum / Count)
        # 0으로 나누는 것 방지 (Count가 0이면 1로 취급, 어차피 Sum도 0임)
        current_count = np.maximum(self.action_count[:steps], 1.0)
        averaged_action = self.action_sum[:steps] / current_count
        
        # 2. 버퍼 Shift (시간 흐름 적용)
        # steps만큼 사용했으니, 뒤에 있는 미래 예측들을 앞으로 당겨옴
        self.action_sum[:-steps] = self.action_sum[steps:]
        self.action_count[:-steps] = self.action_count[steps:]
        
        # 3. 뒤쪽 빈 공간 0으로 초기화
        self.action_sum[-steps:] = 0.0
        self.action_count[-steps:] = 0.0
        
        return averaged_action


DISPLAY = True

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_crop', is_flag=True, default=False)
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')

def main(input, output,
    match_dataset, match_episode, match_camera,
    vis_camera_idx, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    sim_fov, camera_intrinsics, robot_config,
    mirror_crop):

    # Load configuration from YAML file
    config_path = os.path.expanduser(robot_config)
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    robot_config_dict = config_data['robot']
    gripper_config_dict = config_data['gripper']
    camera_config_dict = config_data['camera']
    env_config_dict = config_data['env']

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    # Define movement steps for keyboard control
    pos_step = 0.01          # 1 cm
    rot_step = np.deg2rad(5) # 5 degrees
    gripper_step = 5.0       # 5 mm

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    # defalut not use fisheye lens
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            UmiEnv(
                output_dir=output,
                robot_config=robot_config_dict,
                gripper_config=gripper_config_dict,
                camera_config=camera_config_dict,
                env_config=env_config_dict,
                frequency=frequency,
                obs_image_resolution=obs_res,
                shm_manager=shm_manager
            ) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break
                        # img = VideoFileClip(str(match_video_path)).get_frame(0)

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)

            n_action_steps = cfg.task.shape_meta.action.shape[0] # 혹은 하드코딩 (예: 16)
            ensembler = TemporalEnsembler(max_steps=50, action_dim=7)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = [np.concatenate([
                        obs['robot0_eef_pos'][-1], 
                        obs['robot0_eef_rot_axis_angle'][-1]
                    ])]
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7
                del result

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['ActualTCPPose']
                gripper_state = env.get_gripper_state()
                target_gripper_pos_mm = gripper_state['gripper_position_mm'].copy()
                t_start = time.monotonic()
                iter_idx = 0

                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # # visualize => match_episode first frame ghost view
                    # episode_id = env.replay_buffer.n_episodes
                    # vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    # match_episode_id = episode_id
                    # if match_episode is not None:
                    #     match_episode_id = match_episode
                    # if match_episode_id in episode_first_frame_map:
                    #     match_img = episode_first_frame_map[match_episode_id]
                    #     ih, iw, _ = match_img.shape
                    #     oh, ow, _ = vis_img.shape
                    #     tf = get_image_transform(
                    #         input_res=(iw, ih), 
                    #         output_res=(ow, oh), 
                    #         bgr_to_rgb=False)
                    #     match_img = tf(match_img).astype(np.float32) / 255
                    #     vis_img = (vis_img + match_img) / 2
                    # obs_img = obs['camera0_rgb'][-1]
                    # if mirror_crop:
                    #     crop_img = obs['camera0_rgb_mirror_crop'][-1]
                    #     vis_img = np.concatenate([obs_img, crop_img, vis_img], axis=1)
                    # else:
                    #     vis_img = np.concatenate([obs_img, vis_img], axis=1)
                    
                    # text = f'Episode: {episode_id}'
                    # cv2.putText(
                    #     vis_img,
                    #     text,
                    #     (10,20),
                    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #     fontScale=0.5,
                    #     lineType=cv2.LINE_AA,
                    #     thickness=3,
                    #     color=(0,0,0)
                    # )
                    # cv2.putText(
                    #     vis_img,
                    #     text,
                    #     (10,20),
                    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #     fontScale=0.5,
                    #     thickness=1,
                    #     color=(255,255,255)
                    # )
                    # cv2.imshow('default', vis_img[...,::-1])
                    # _ = cv2.pollKey()

                    dpos = np.zeros(3)
                    drot_xyz = np.zeros(3)
                    d_gripper = 0

                    press_events = key_counter.get_press_events()
                    # Ensure compatibility: Convert string inputs to KeyCode objects
                    press_events = [KeyCode(char=k) if isinstance(k, str) and len(k)==1 else k for k in press_events]

                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        # elif key_stroke == KeyCode(char='e'):
                        #     # Next episode
                        #     if match_episode is not None:
                        #         match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        # elif key_stroke == KeyCode(char='w'):
                        #     # Prev episode
                        #     if match_episode is not None:
                        #         match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)
                            pos = ep['robot0_eef_pos'][0]
                            rot = ep['robot0_eef_rot_axis_angle'][0]
                            grip = ep['robot0_gripper_width'][0]
                            start_pose = np.concatenate([pos, rot])
                            start_grip = grip[0]
                            env.robot.servoL(start_pose, duration=duration)
                            env.gripper.schedule_waypoint(start_grip, target_time=time.time() + duration)
                            time.sleep(duration)
                            target_pose = start_pose
                            gripper_target_pos = start_grip
                        # elif key_stroke == Key.backspace:
                        #     if click.confirm('Are you sure to drop an episode?'):
                        #         env.drop_episode()
                        #         key_counter.clear()

                        # Teleoperation Key Mappings
                        elif key_stroke == KeyCode(char='w'):
                            dpos[0] += pos_step
                        elif key_stroke == KeyCode(char='s'):
                            dpos[0] -= pos_step
                        elif key_stroke == KeyCode(char='a'):
                            dpos[1] += pos_step
                        elif key_stroke == KeyCode(char='d'):
                            dpos[1] -= pos_step
                        elif key_stroke == KeyCode(char='r'):
                            dpos[2] += pos_step
                        elif key_stroke == KeyCode(char='f'):
                            dpos[2] -= pos_step
                        elif key_stroke == KeyCode(char='y'):
                            drot_xyz[0] += rot_step
                        elif key_stroke == KeyCode(char='i'):
                            drot_xyz[0] -= rot_step
                        elif key_stroke == KeyCode(char='h'):
                            drot_xyz[1] += rot_step
                        elif key_stroke == KeyCode(char='k'):
                            drot_xyz[1] -= rot_step
                        elif key_stroke == KeyCode(char='u'):
                            drot_xyz[2] += rot_step
                        elif key_stroke == KeyCode(char='j'):
                            drot_xyz[2] -= rot_step
                        elif key_stroke == KeyCode(char='o'):
                            d_gripper += gripper_step
                        elif key_stroke == KeyCode(char='p'):
                            d_gripper -= gripper_step
                    # policy start flag
                    if start_policy:
                        break
                    
                    target_pose[:3] += dpos
                    
                    current_rot = st.Rotation.from_rotvec(target_pose[3:])
                    delta_rot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[3:] = (current_rot * delta_rot).as_rotvec()
                    
                    max_gripper_width = gripper_config_dict.get('gripper_max_width_mm', 80.0)
                    target_gripper_pos_mm = np.clip(target_gripper_pos_mm + d_gripper, 0.0, max_gripper_width)

                    action = np.zeros((7,))
                    action[:6] = target_pose
                    action[-1] = target_gripper_pos_mm     

                    if DISPLAY:
                        current_width_mm = obs['robot0_gripper_width'][-1]
                        print(f"✊ [Gripper] Current(State): {current_width_mm.item():.2f} mm  -->  Target(Action): {target_gripper_pos_mm.item():.2f} mm")

                    # execute teleop command
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 2.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # Fix: Force inject start pose to prevent IndexError
                    raw_state = env.get_robot_state()
                    current_pose = raw_state['ActualTCPPose'] # (6,) [x,y,z,rx,ry,rz]
                    env.episode_start_pose = current_pose

                    time.sleep(0.1) 
                    
                    obs = env.get_obs()
                    episode_start_pose = [np.concatenate([
                        obs['robot0_eef_pos'][-1], 
                        obs['robot0_eef_rot_axis_angle'][-1]
                    ])]

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        obs_latency = time.time() - obs_timestamps[-1]
                        print(f'Obs latency {obs_latency}')

                        # ===============================================================
                        if DISPLAY:
                            # [Debug] Print raw observation details (excluding images)
                            print("\n🔎 === Raw Observation Details (No Images) ===")
                            for key, value in obs.items():
                                if hasattr(value, 'shape'):
                                    if 'rgb' in key or len(value.shape) > 2:
                                        print(f"📷 Key: {key:<35} | 📐 Shape: {value.shape} (Image skipped)")
                                    else:
                                        print(f"🔢 Key: {key:<35} | 📐 Shape: {value.shape}")
                                        with np.printoptions(precision=4, suppress=True):
                                            print(f"   👉 Latest: {value[-1]}")
                                else:
                                    print(f"📝 Key: {key:<35} | Value: {value}")
                            print("====================================================\n")
                        # ===============================================================

                        # =================================================================
                        # [추가됨] Observation 역보정 (Inverse Mapping)
                        # 목적: Policy가 학습했던 분포(Scale)와 맞추기 위함
                        # 방법: Action에 썼던 보간법의 입력(xp)과 출력(fp)을 바꿔서 적용!
                        # =================================================================
                        
                        # Action 때 정의했던 기준점 그대로 가져옴 (단위: mm)
                        # xp: 모델 기준 (Original Scale)
                        # map_xp = [0.0, 75.0, 77.0, 88.0]
                        # # fp: 실제 로봇 기준 (Real Scale)
                        # map_fp = [0.0, 70.0, 77.0, 88.0]

                        # # 현재 관측된 그리퍼 값 (단위: mm)
                        # # 주의: obs['robot0_gripper_width']는 배열일 수 있으니 통째로 넘깁니다.
                        # real_gripper_val = obs['robot0_gripper_width']

                        # # [핵심] np.interp(x, xp, fp) 순서에서 xp와 fp를 바꿉니다!
                        # # 입력(x): 실제 로봇 값 (real_gripper_val)
                        # # 기준(xp): map_fp (실제값 기준)
                        # # 목표(fp): map_xp (모델값 기준)
                        # obs['robot0_gripper_width'] = np.interp(
                        #     real_gripper_val, 
                        #     map_fp,  # <--- 여기가 입력 기준 (X축)
                        #     map_xp   # <--- 여기가 출력 목표 (Y축)
                        # )

                        # (디버깅용) 혹시 차이가 얼마나 나는지 보고 싶으면 주석 해제하세요
                        # if DISPLAY:
                        #     diff = obs['robot0_gripper_width'][-1] - real_gripper_val[-1]
                        #     print(f"👀 [Obs Hook] Real: {real_gripper_val[-1]:.2f} -> Fake: {obs['robot0_gripper_width'][-1]:.2f} (Diff: {diff:.2f})")

                        # =================================================================

                        # Convert gripper unit: mm -> m
                        for i in range(len(robot_config)):
                            key = f'robot{i}_gripper_width'
                            if key in obs:
                                obs[key] = obs[key] / 1000.0

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            # Scale action: m -> mm
                            action[..., 6::7] = action[..., 6::7] * 1000.0
                            inference_latency = time.time() - s
                            print('Inference latency:', inference_latency)

                            # =================================================================
                            
                            # 1. 새로운 예측값(Chunk)을 앙상블 버퍼에 투표(Update)
                            ensembler.update(action)
                            
                            # 2. 이번 주기(steps_per_inference)에 실행할 만큼만 꺼내오기(Get & Shift)
                            # 중요: 여기서 꺼내온 액션이 바로 '평균'이 적용된 부드러운 액션입니다.
                            final_action = ensembler.get_next_actions(steps_per_inference)
                            
                            # 변수명 교체 (action -> final_action)
                            # 이제부터 아래 로직은 이 '섞인 액션'을 사용합니다.
                            action = final_action 

                            # (디버깅용) 앙상블 잘 됐나 확인
                            if DISPLAY:
                                print(f"✨ Ensembled Action Shape: {action.shape}")

                            # =================================================================
                            
                            # =================================================================
                            # [수정됨] 78 이상 1:1 추종 (Linear Extension)
                            # 1. 0 ~ 75 구간: 모델보다 좀 덜 열림 (75 -> 70)
                            # 2. 75 ~ 78 구간: 빠르게 따라잡음 (78 -> 78)
                            # 3. 78 이상 구간: 모델 값 그대로 따라감 (1:1)
                            # =================================================================
                            
                            # [입력(Model) 기준점]
                            # 0(닫힘) --- 75(억제) --- 78(동기화) --- 85(최대개방)
                            # xp = [0.0, 75.0, 77.0, 88.0]
                            
                            # # [출력(Robot) 목표점]
                            # # 0(닫힘) --- 70(실제값) --- 78(실제값) --- 85(실제값)
                            # fp = [0.0, 70.0, 77.0, 88.0]
                            
                            # # 작동 원리:
                            # # 1. 모델 75 -> 실제 70
                            # # 2. 모델 78 -> 실제 78 (여기서 만남)
                            # # 3. 모델 80 -> 실제 80 (78과 85 사이를 1:1로 연결했으므로 그대로 나옴)
                            # # 4. 모델 85 -> 실제 85

                            # action[..., 6::7] = np.interp(
                            #     action[..., 6::7], 
                            #     xp, 
                            #     fp
                            # )

                             # ===============================================================
                            # [Debug] Display detailed coordinate transformation
                            # ===============================================================
                            if DISPLAY:
                                print("\n" + "="*70)
                                print("🔬 === Coordinate Transformation Analysis ===")
                                print("="*70)
                                
                                # 1. Raw Absolute Coordinates
                                print("\n📍 [Step 1] Raw Absolute Coordinates (Base Frame)")
                                print("-"*50)
                                raw_pos = obs['robot0_eef_pos'][-1]
                                raw_rot = obs['robot0_eef_rot_axis_angle'][-1]
                                raw_gripper = obs['robot0_gripper_width'][-1]
                                with np.printoptions(precision=6, suppress=True):
                                    print(f"  robot0_eef_pos (Abs):            {raw_pos}")
                                    print(f"  robot0_eef_rot_axis_angle (Abs): {raw_rot}")
                                    print(f"  robot0_gripper_width(m):        {raw_gripper}")
                            
                                # 2. Episode Start Pose
                                print("\n📍 [Step 2] Episode Start Pose (Reference)")
                                print("-"*50)
                                with np.printoptions(precision=6, suppress=True):
                                    print(f"  episode_start_pose[0][:3] (Pos): {episode_start_pose[0][:3]}")
                                    print(f"  episode_start_pose[0][3:] (Rot): {episode_start_pose[0][3:]}")
                                
                                # 3. Manual Relative Calculation
                                print("\n📍 [Step 3] Manual Relative Calculation (Validation)")
                                print("-"*50)
                                
                                current_pose_6d = np.concatenate([raw_pos, raw_rot])
                                current_mat = pose_to_mat(current_pose_6d)
                                start_mat = pose_to_mat(episode_start_pose[0])
                                
                                # Relative = start^(-1) @ current
                                rel_mat = np.linalg.inv(start_mat) @ current_mat
                                rel_pos_manual = rel_mat[:3, 3]
                                
                                with np.printoptions(precision=6, suppress=True):
                                    print(f"  Relative Pos (Calculated): {rel_pos_manual}")
                                    print(f"  Simple Diff (pos - start): {raw_pos - episode_start_pose[0][:3]}")

                                print("\n📍 [Step 4] get_real_umi_obs_dict() Result")
                                print("-"*50)
                                for key, value in obs_dict_np.items():
                                    if hasattr(value, 'shape'):
                                        if 'rgb' in key:
                                            print(f"  {key:<45} | Shape: {value.shape} (Image)")
                                        else:
                                            print(f"  {key:<45} | Shape: {value.shape}")
                                            with np.printoptions(precision=6, suppress=True):
                                                if len(value.shape) == 1:
                                                    print(f"    👉 Value: {value}")
                                                else:
                                                    print(f"    👉 Latest: {value[-1]}")
                                
                                print("\n" + "="*70)

                                print("\n🎯 === Policy Output (Raw Action - Relative) ===")
                                print(f"  raw_action shape: {raw_action.shape}")
                                with np.printoptions(precision=6, suppress=True):
                                    print(f"  raw_action[0] (First Step):")
                                    print(f"    Pos (0:3):      {raw_action[0, :3]}")
                                    if raw_action.shape[-1] >= 9:
                                        print(f"    Rot (3:9):      {raw_action[0, 3:9]}")
                                    if raw_action.shape[-1] >= 10:
                                        print(f"    Gripper (9:10): {raw_action[0, 9:]}")

                                # target_width_mm = action[0, 6]
                                # print(f"✊ [Gripper] Current(State): {current_width_mm.item():.2f} mm  -->  Target(Action): {target_width_mm.item():.2f} mm")

                                print("\n🎯 === Final Action (Absolute Coords) ===")
                                print(f"  action shape: {action.shape}")
                                with np.printoptions(precision=6, suppress=True):
                                    print(f"  action[0] (First Step):")
                                    print(f"    Pos (0:3):    {action[0, :3]}")
                                    print(f"    Rot (3:6):    {action[0, 3:6]}")
                                    print(f"    Gripper (6):  {action[0, 6]}")
                                
                                delta_pos = action[0, :3] - raw_pos
                                with np.printoptions(precision=6, suppress=True):
                                    print(f"\n  📐 Delta (Action - Current):")
                                    print(f"    delta_pos: {delta_pos}")
                                    print(f"    Dist:      {np.linalg.norm(delta_pos):.6f}m")
                                
                                print("\n" + "="*70)
                                print("🔬 === Analysis Complete ===")
                                print("="*70 + "\n")

                                print("\n🎯 === Final Action (Absolute Coords) ===")
                                print(f"  action shape: {action.shape}")
                                with np.printoptions(precision=6, suppress=True):
                                    print(f"  action[0] (First Step):")
                                    print(f"    Pos (0:3):    {action[0, :3]}")
                                    print(f"    Rot (3:6):    {action[0, 3:6]}")
                                    print(f"    Gripper (6):  {action[0, 6]}")
                                
                                delta_pos = action[0, :3] - raw_pos
                                with np.printoptions(precision=6, suppress=True):
                                    print(f"\n  📐 Delta (Action - Current):")
                                    print(f"    delta_pos: {delta_pos}")
                                    print(f"    Dist:      {np.linalg.norm(delta_pos):.6f}m")
                                
                                print("\n" + "="*70)
                                print("🔬 === Analysis Complete ===")
                                print("="*70 + "\n")
                            # ====================================================================
                        

                        # convert policy action to env actions
                        this_target_poses = action
                        # this_target_poses[:,2] = np.maximum(this_target_poses[:,2], 0.055)

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # # visualize
                        # episode_id = env.replay_buffer.n_episodes
                        # if mirror_crop:
                        #     vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        #     crop_img = obs['camera0_rgb_mirror_crop'][-1]
                        #     vis_img = np.concatenate([vis_img, crop_img], axis=1)
                        # else:
                        #     vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        # text = 'Episode: {}, Time: {:.1f}'.format(
                        #     episode_id, time.monotonic() - t_start
                        # )
                        # cv2.putText(
                        #     vis_img,
                        #     text,
                        #     (10,20),
                        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #     fontScale=0.5,
                        #     thickness=1,
                        #     color=(255,255,255)
                        # )
                        # cv2.imshow('default', vis_img[...,::-1])

                        # _ = cv2.pollKey()

                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        # wakeup early
                        precise_wait(t_cycle_end - obs_latency - inference_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()