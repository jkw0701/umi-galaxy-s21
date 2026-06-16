# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
# from umi.real_world.spacemouse_shared_memory import Spacemouse 
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.common.precise_sleep import precise_wait
from umi.common.latency_util import get_latency
from matplotlib import pyplot as plt

@click.command()
@click.option('-rh', '--robot_hostname', default='192.168.1.10')
@click.option('-f', '--frequency', type=float, default=30)
def main(robot_hostname, frequency):
    dt = 1/frequency
    
    # [설정] 명령 도달 여유 시간 (중요!)
    action_latency = 0.1
    
    duration = 10.0
    period = 2.0
    
    # [설정] 움직임 진폭 설정
    amp_pos = 0.05  # 위치: 5cm 왔다갔다
    amp_rot = 0.1   # 회전: 약 0.1라디안 (약 5.7도) 왔다갔다

    with SharedMemoryManager() as shm_manager:
        with FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            frequency=200,
            Kx_scale=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
            Kxd_scale=np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
            verbose=False
        ) as controller:
            print('Waiting for robot...')
            time.sleep(1.0) 
            assert controller.is_ready, "Robot is NOT ready!"

            print('Ready! Robot will move ALL AXES (X,Y,Z, Rx,Ry,Rz).')
            print('⚠️ WARNING: Robot will dance! Keep space clear! (공간 확보 필수)')
            
            # 현재 위치 저장 (기준점)
            state = controller.get_state()
            start_pose = state['ActualTCPPose'].copy() 
            
            t_start = time.time()
            
            t_target_list = list()
            x_target_list = list()
            
            # === 자동 움직임 루프 ===
            iter_idx = 0
            while True:
                t_now = time.time()
                t_elapsed = t_now - t_start
                
                if t_elapsed > duration:
                    break

                # 타이밍 계산
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_command_target = t_cycle_end + action_latency

                # [핵심 변경] 모든 축에 대한 사인파 계산
                # 위상(Phase)을 조금씩 다르게 줘서 서로 다르게 움직이는 것처럼 보이게 함 (멋짐 추가)
                
                sine_base = np.sin(t_elapsed * 2 * np.pi / period)
                
                target_pose = start_pose.copy()

                # 1. 위치 (Translation) X, Y, Z
                target_pose[0] += sine_base * amp_pos           # X축
                target_pose[1] += np.cos(t_elapsed) * amp_pos   # Y축 (코사인으로 엇박자)
                target_pose[2] += sine_base * amp_pos           # Z축

                # 2. 회전 (Rotation) Rx, Ry, Rz
                # 회전은 너무 크면 위험하므로 amp_rot 사용
                target_pose[3] += sine_base * amp_rot           # Rx
                target_pose[4] += np.cos(t_elapsed) * amp_rot   # Ry
                target_pose[5] += sine_base * amp_rot           # Rz

                # 데이터 기록
                t_target_list.append(t_command_target)
                x_target_list.append(target_pose.copy())

                # 명령 전송
                controller.schedule_waypoint(target_pose, t_command_target)
                
                # 주기 맞추기
                precise_wait(t_cycle_end, time_func=time.time)
                iter_idx += 1
                
                if iter_idx % 30 == 0:
                    print(f"Moving... Time: {t_elapsed:.1f}s")

            print("Movement finished. Collecting data...")
            time.sleep(1.0) 
            states = controller.get_all_state()

    # 데이터 정리
    t_target = np.array(t_target_list)
    x_target = np.array(x_target_list)
    
    t_actual = states['robot_receive_timestamp']
    x_actual = states['ActualTCPPose']
    
    print(f"Collected: Target {len(t_target)}, Actual {len(t_actual)}")

    # 그래프 그리기 (6개 축 모두 표시)
    n_dims = 6
    fig, axes = plt.subplots(n_dims, 3)
    fig.set_size_inches(15, 20, forward=True) # 세로로 좀 더 길게 늘림

    for i in range(n_dims):
        latency, info = get_latency(x_target[...,i], t_target, x_actual[...,i], t_actual)
        
        # 축 이름 설정
        axis_name = ["X", "Y", "Z", "Rx", "Ry", "Rz"][i]
        print(f"Axis {axis_name} ({i}) Latency: {latency:.4f} sec")

        row = axes[i]
        
        # 1. Cross Correlation
        ax = row[0]
        ax.plot(info['lags'], info['correlation'])
        ax.set_title(f"[{axis_name}] Cross Corr (Lat={latency:.4f}s)")

        # 2. Raw Observation
        ax = row[1]
        ax.plot(t_target, x_target[...,i], label='target')
        ax.plot(t_actual, x_actual[...,i], label='actual')
        ax.legend()
        ax.set_title(f"[{axis_name}] Raw Trajectory")

        # 3. Aligned
        ax = row[2]
        t_samples = info['t_samples'] - info['t_samples'][0]
        ax.plot(t_samples, info['x_target'], label='target')
        ax.plot(t_samples-latency, info['x_actual'], label='actual-aligned')
        ax.legend()
        ax.set_title(f"[{axis_name}] Latency Corrected")

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()