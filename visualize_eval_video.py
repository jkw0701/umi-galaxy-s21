import pickle
import cv2
import numpy as np
import matplotlib
# 서버 환경(GUI 없음)을 위해 Agg 백엔드 설정
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import sys
from tqdm import tqdm
import glob
from pathlib import Path

# ==================================================================================
# [설정] 터미널에서 인자로 받거나, 기본값 사용
# ==================================================================================
# 사용법:
#   python visualize_eval_video.py                              → DEFAULT_EPISODES 처리
#   python visualize_eval_video.py <root>                        → DEFAULT_EPISODES 처리
#   python visualize_eval_video.py <root> ep1 ep2 ...            → 지정한 episode 폴더만 처리
TARGET_ROOT_DIR = sys.argv[1] if len(sys.argv) > 1 else '/home/kist/Umi_ws/universal_manipulation_interface/eval_data'

# 결과물을 모아둘 폴더 이름 (TARGET_ROOT_DIR 아래에 생성됨)
RESULT_FOLDER_NAME = 'all_analysis_results/test'

# 인자 없을 때 기본 처리할 episode 폴더 이름 목록
DEFAULT_EPISODES = ['211_fail_intermediate', '208']
# ==================================================================================

def render_plot(fig, axes, step_log, step_idx, global_traj, all_logs):
    """ X, Y, Z (위치) + Rx, Ry, Rz (회전 axis-angle) + Grip (gripper width) 모두를 그리는 함수 """
    full_chunk = step_log['processed_action_chunk']   # (T, >=7): pos(3)+rot(3)+grip(1)
    is_new_mask = step_log['is_new_mask']

    obs_timestamp = step_log['timestamp'] - step_log['obs_latency']
    saved_timestamps = step_log['action_timestamps']
    if len(saved_timestamps) > 1:
        dt = saved_timestamps[1] - saved_timestamps[0]
    else:
        dt = 0.1
    full_timestamps = np.arange(len(full_chunk)) * dt + obs_timestamp

    global_times = global_traj['times']
    start_time = global_times[0]
    global_x_time = global_times - start_time

    # dim 0-2: position(m),  dim 3-5: rotation axis-angle(rad),  dim 6: gripper width(m)
    dim_names    = ['X',      'Y',      'Z',      'Rx',         'Ry',         'Rz',         'Grip']
    dim_units    = ['m',      'm',      'm',      'rad',        'rad',        'rad',        'mm']
    actual_color = ['black',  'black',  'black',  'darkgreen',  'darkgreen',  'darkgreen',  'darkviolet']
    chunk_color  = ['cyan',   'cyan',   'cyan',   'coral',      'coral',      'coral',      'plum']
    active_color = ['magenta','magenta','magenta','orangered',  'orangered',  'orangered',  'purple']
    local_hw     = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 10.0]

    # rotation actual 데이터가 없으면(모두 0) global actual 선 표시 생략
    rot_available = not np.all(global_traj['rot'] == 0)

    for dim_idx in range(7):
        is_rot  = 3 <= dim_idx < 6
        is_grip = dim_idx == 6

        if is_grip:
            global_ref = global_traj['grip']
        elif is_rot:
            global_ref = global_traj['rot'][:, dim_idx - 3]
        else:
            global_ref = global_traj['pos'][:, dim_idx]

        # ── Global View ──────────────────────────────────────────────────────
        ax_global = axes[dim_idx, 0]
        ax_global.clear()

        # actual 선: 위치/그리퍼는 항상, 회전은 데이터 있을 때만
        if not is_rot or rot_available:
            ax_global.plot(global_x_time, global_ref,
                           color=actual_color[dim_idx], linewidth=1.5,
                           label='Robot actual', alpha=0.7, zorder=1)

        # 모든 step의 action chunk (모델 예측)
        _chunk_label_added = False
        _active_label_added = False
        stride = 2
        for i in range(0, len(all_logs), stride):
            log = all_logs[i]
            chunk_act = log['processed_action_chunk'][:, dim_idx]
            c_obs_time = log['timestamp'] - log['obs_latency']
            c_ts = np.arange(len(chunk_act)) * dt + c_obs_time
            chunk_x_time = c_ts - start_time

            lbl_chunk = 'Model chunk (all)' if not _chunk_label_added else None
            ax_global.plot(chunk_x_time, chunk_act,
                           color=chunk_color[dim_idx], alpha=0.35, linewidth=1,
                           label=lbl_chunk, zorder=2)
            if lbl_chunk:
                _chunk_label_added = True

            c_mask = log['is_new_mask']
            if c_mask is not None and np.any(c_mask):
                lbl_active = 'Active chunk (실행됨)' if not _active_label_added else None
                ax_global.plot(chunk_x_time[c_mask], chunk_act[c_mask],
                               color=active_color[dim_idx], linewidth=1.8, alpha=0.7,
                               label=lbl_active, zorder=3)
                if lbl_active:
                    _active_label_added = True

        # 현재 시각 마커
        current_time_rel = step_log['timestamp'] - start_time
        curr_idx = min(np.searchsorted(global_times, step_log['timestamp']), len(global_ref) - 1)
        ax_global.axvline(x=current_time_rel, color='gray', linestyle=':', alpha=0.5,
                          label='Current time')
        if not is_rot or rot_available:
            ax_global.plot(global_x_time[curr_idx], global_ref[curr_idx],
                           marker='o', color='red', markersize=7, zorder=10,
                           label='Robot pos now')

        ax_global.set_ylabel(f"{dim_names[dim_idx]} ({dim_units[dim_idx]})", fontsize=7)
        if dim_idx == 0:
            ax_global.set_title("Position — Global View", fontsize=9, fontweight='bold')
        elif dim_idx == 3:
            ax_global.set_title("Rotation (axis-angle) — Global View", fontsize=9, fontweight='bold')
        elif dim_idx == 6:
            ax_global.set_title("Gripper — Global View", fontsize=9, fontweight='bold')
        if dim_idx not in (2, 5, 6):
            ax_global.set_xticklabels([])
        else:
            ax_global.set_xlabel("Time (s)", fontsize=7)
        ax_global.grid(True, linestyle=':')
        ax_global.legend(fontsize=5, loc='upper left', framealpha=0.6)

        # ── Local View ───────────────────────────────────────────────────────
        ax_local = axes[dim_idx, 1]
        ax_local.clear()

        chunk_indices_local = np.arange(len(full_chunk))

        # local actual: 위치/그리퍼는 interp, 회전은 없으면 생략
        if not is_rot or rot_available:
            full_actual_path = np.interp(full_timestamps, global_times, global_ref)
        else:
            full_actual_path = None

        mask = is_new_mask
        if mask is None or len(mask) == 0:
            mask = np.zeros(len(full_chunk), dtype=bool)

        # 이전 chunk 부분 (재사용, 흐릿하게)
        if np.any(~mask):
            ax_local.plot(chunk_indices_local[~mask], full_chunk[~mask, dim_idx],
                          color='steelblue', linestyle='--', alpha=0.35,
                          label='Model chunk (재사용)')
            if full_actual_path is not None:
                ax_local.plot(chunk_indices_local[~mask], full_actual_path[~mask],
                              color='gray', linestyle='-', marker='x', markersize=3,
                              alpha=0.4, label='Robot actual (interp)')

        # 새로 실행되는 chunk 부분
        if np.any(mask):
            ax_local.plot(chunk_indices_local[mask], full_chunk[mask, dim_idx],
                          color='blue', linestyle='--', linewidth=1.5, alpha=0.8,
                          label='Model chunk (신규)')
            if full_actual_path is not None:
                ax_local.plot(chunk_indices_local[mask], full_actual_path[mask],
                              color='lime', linewidth=2, marker='o', markersize=3,
                              label='Robot actual (신규)')

            first_idx = np.where(mask)[0][0]
            if first_idx > 0:
                ax_local.plot([first_idx-1, first_idx],
                              full_chunk[[first_idx-1, first_idx], dim_idx],
                              color='blue', linestyle=':', alpha=0.3)

        if dim_idx == 0:
            ax_local.set_title("Position — Local Chunk", fontsize=9, fontweight='bold')
        elif dim_idx == 3:
            ax_local.set_title("Rotation — Local Chunk", fontsize=9, fontweight='bold')
        elif dim_idx == 6:
            ax_local.set_title("Gripper — Local Chunk", fontsize=9, fontweight='bold')

        center_val = full_chunk[0, dim_idx]
        ax_local.set_ylim(center_val - local_hw[dim_idx], center_val + local_hw[dim_idx])
        ax_local.grid(True)
        ax_local.legend(fontsize=5, loc='upper left', framealpha=0.6)
        if dim_idx not in (2, 5, 6):
            ax_local.set_xticklabels([])

    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    image = np.asarray(canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(int(height), int(width), 4)[..., :3]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def render_dashboard(height, width, step_log):
    """ X, Y, Z 값을 모두 표시하는 대시보드 (기존과 동일) """
    dashboard = np.zeros((height, width, 3), dtype=np.uint8)

    pos = step_log.get('robot_eef_pos', np.zeros(3))
    gripper_m = step_log.get('robot_gripper_width', 0.0)
    if hasattr(gripper_m, 'item'): gripper_m = gripper_m.item()
    gripper_mm = gripper_m * 1000.0
    latency = step_log.get('obs_latency', 0.0) * 1000.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    y = 40
    step = 35

    cv2.putText(dashboard, "STATUS MONITOR", (20, y), font, 0.7, white, 2)
    y += step
    cv2.putText(dashboard, f"Latency: {latency:.1f} ms", (20, y), font, 0.7, yellow, 2)
    y += step + 10

    cv2.putText(dashboard, "Robot Position (m)", (20, y), font, 0.6, green, 1)
    y += step
    cv2.putText(dashboard, f"X: {pos[0]:.4f}", (30, y), font, 0.8, white, 2)
    y += step
    cv2.putText(dashboard, f"Y: {pos[1]:.4f}", (30, y), font, 0.8, white, 2)
    y += step
    cv2.putText(dashboard, f"Z: {pos[2]:.4f}", (30, y), font, 0.8, (0,0,255), 2)
    y += step + 10

    cv2.putText(dashboard, "Gripper Width", (20, y), font, 0.6, green, 1)
    y += step
    bar_max = width - 60
    bar_cur = int((gripper_mm / 85.0) * bar_max)
    cv2.rectangle(dashboard, (30, y-15), (30 + bar_max, y+15), (50, 50, 50), -1)
    cv2.rectangle(dashboard, (30, y-15), (30 + bar_cur, y+15), (255, 100, 0), -1)
    cv2.putText(dashboard, f"{gripper_mm:.2f} mm", (40, y+5), font, 0.6, white, 2)

    return dashboard

def process_single_video(video_path, log_path, output_path):
    """ 하나의 에피소드를 처리하는 함수 """
    print(f"\n🚀 Processing: {video_path}")

    if not os.path.exists(log_path):
        print(f"   ⚠️ Log file not found, skipping: {log_path}")
        return

    with open(log_path, 'rb') as f:
        logs = pickle.load(f)

    all_times = [l['timestamp'] for l in logs]
    all_pos   = [l['robot_eef_pos'] for l in logs]
    all_rot   = [l.get('robot_eef_rot_axis_angle', np.zeros(3)) for l in logs]
    # gripper_width: actual 은 m 로 저장되지만 chunk 의 gripper 컬럼은 mm 라서 mm 로 통일
    all_grip  = [float(np.asarray(l.get('robot_gripper_width', 0.0)).reshape(-1)[0]) * 1000.0 for l in logs]
    global_traj = {
        'times': np.array(all_times),
        'pos':   np.array(all_pos),
        'rot':   np.array(all_rot),
        'grip':  np.array(all_grip),
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fig, axes = plt.subplots(7, 2, figsize=(10, 21), dpi=80)
    plt.tight_layout()

    # Init
    dummy_graph = render_plot(fig, axes, logs[0], 0, global_traj, logs)
    scale = video_h / dummy_graph.shape[0]
    graph_w = int(dummy_graph.shape[1] * scale)
    dashboard_w = 300

    final_w = video_w + dashboard_w + graph_w
    final_h = video_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (final_w, final_h))

    last_log_idx = -1
    cached_graph = None
    cached_dashboard = None

    for current_frame in tqdm(range(total_frames), desc="   Rendering"):
        ret, frame = cap.read()
        if not ret: break

        current_time = logs[0]['timestamp'] + (current_frame / fps)
        log_idx = 0
        for i, log in enumerate(logs):
            if log['timestamp'] > current_time:
                break
            log_idx = i

        if log_idx != last_log_idx or cached_graph is None:
            raw_graph = render_plot(fig, axes, logs[log_idx], log_idx, global_traj, logs)
            cached_graph = cv2.resize(raw_graph, (graph_w, video_h))
            cached_dashboard = render_dashboard(video_h, dashboard_w, logs[log_idx])
            last_log_idx = log_idx

        combined = np.hstack((frame, cached_dashboard, cached_graph))
        cv2.putText(combined, f"Frame: {current_frame}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(combined)

    cap.release()
    out.release()
    plt.close(fig)
    print(f"   ✅ Saved: {output_path}")

def main():
    target_root = Path(TARGET_ROOT_DIR)
    videos_dir = target_root / 'videos'

    if not videos_dir.exists():
        print(f"❌ Error: 'videos' folder not found in {target_root}")
        print("Please check the path. It should contain a 'videos' subdirectory.")
        return

    results_dir = target_root / RESULT_FOLDER_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Results will be saved in: {results_dir}")

    # 처리할 episode 폴더 결정: CLI 두 번째 이후 인자가 있으면 그걸 사용, 없으면 DEFAULT_EPISODES
    if len(sys.argv) > 2:
        episode_names = sys.argv[2:]
    else:
        episode_names = DEFAULT_EPISODES

    episode_dirs = []
    for name in episode_names:
        ep_path = videos_dir / name
        if not ep_path.is_dir():
            print(f"⚠️  Episode folder not found, skipping: {ep_path}")
            continue
        episode_dirs.append(ep_path)

    if not episode_dirs:
        print("❌ No valid episode folders to process.")
        return

    print(f"📂 Processing {len(episode_dirs)} episode(s): {[d.name for d in episode_dirs]}")

    for ep_dir in episode_dirs:
        log_path = ep_dir / 'debug_log.pkl'

        video_candidates = list(ep_dir.glob('*.mp4'))
        source_video = None
        for v in video_candidates:
            if "analysis_result" not in v.name:
                source_video = v
                break

        if source_video and log_path.exists():
            episode_num = ep_dir.name
            output_filename = f"result_ep_{episode_num}.mp4"
            output_path = results_dir / output_filename
            process_single_video(str(source_video), str(log_path), str(output_path))
        else:
            print(f"⚠️  Skipping {ep_dir.name}: missing log or video")

    print(f"\n✨ All jobs finished! Check folder: {results_dir}")

if __name__ == "__main__":
    main()
