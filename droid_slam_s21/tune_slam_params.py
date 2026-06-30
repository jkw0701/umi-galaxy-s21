"""
DROID-SLAM 하이퍼파라미터 자동 튜닝 루프.

탐색 전략:
  - filter_thresh → keyframe_thresh → warmup → frontend_window 순서로 단계적 탐색
  - 한 파라미터에서 개선 없으면 best 조합 복원 후 다음 파라미터로 이동
  - 불안정 0개 또는 max_iter 초과 시 종료

Usage:
    python droid_slam_s21/tune_slam_params.py \
        --session_dir /path/to/session_dir \
        --calibration_dir example/calibration_s21 \
        --ref aruco
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import pathlib
import subprocess
import json
import click
import numpy as np
import pandas as pd


# ── 탐색 범위 (시작값, 스텝, 최대값) ─────────────────────────────────────
PARAM_SCHEDULE = {
    'filter_thresh':   dict(start=2.4, step=0.2,  max=6.0),
    'keyframe_thresh': dict(start=2.8, step=-0.2, max=0.5),  # 2.8부터 낮추며 탐색
    'warmup':          dict(start=8,   step=4,    max=40),
    'frontend_window': dict(start=25,  step=5,    max=70),
}

FIXED_PARAMS = dict(
    frontend_thresh=16.0,
    backend_thresh=22.0,
    frontend_radius=2,
    frontend_nms=1,
    backend_radius=2,
    backend_nms=3,
    beta=0.3,
    buffer=512,
)


def load_z_mm(demo_dir: pathlib.Path):
    csv_path = demo_dir / 'camera_trajectory.csv'
    json_path = demo_dir / 'tx_slam_tag.json'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df_valid = df[df['is_lost'] == False].reset_index(drop=True)
    scale = 1.0
    if json_path.exists():
        tag = json.load(open(json_path))
        scale = float(tag.get('scale_factor', 1.0))
    return df_valid['z'].to_numpy() / scale * 1000


def count_unstable(session_dir: pathlib.Path, start: int, end: int, threshold: float):
    demos_dir = session_dir / 'demos'
    demo_dirs = sorted([d for d in demos_dir.iterdir()
                        if d.is_dir() and d.name.startswith('demo_')])
    unstable = []
    for i, demo_dir in enumerate(demo_dirs):
        z = load_z_mm(demo_dir)
        if z is None:
            continue
        seg = z[start:end+1] if len(z) > end else z[start:]
        if len(seg) < 2:
            continue
        max_diff = float(np.abs(np.diff(seg)).max())
        if max_diff > threshold:
            unstable.append((i, demo_dir.name, max_diff))
    return unstable


def run(cmd, desc):
    print(f"\n>>> {desc}")
    print(f"    {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"[ERROR] {desc} failed (returncode={result.returncode})")
        sys.exit(1)


def run_one_iter(session_dir, script_dir, intrinsics, ref,
                 filter_thresh, keyframe_thresh, warmup, frontend_window, label):
    """SLAM → calibration → dataset plan 한 사이클 실행."""
    run([
        'python', str(script_dir / '03_batch_slam.py'),
        '-i', str(session_dir / 'demos'),
        '--intrinsics', str(intrinsics),
        '--filter_thresh', str(filter_thresh),
        '--keyframe_thresh', str(keyframe_thresh),
        '--warmup', str(warmup),
        '--frontend_window', str(frontend_window),
        '--frontend_thresh', str(FIXED_PARAMS['frontend_thresh']),
        '--backend_thresh', str(FIXED_PARAMS['backend_thresh']),
        '--overwrite',
    ], f"[{label}] DROID-SLAM")

    run([
        'python', str(script_dir / '05_run_calibrations_per_episode.py'),
        '--ref', ref,
        str(session_dir),
    ], f"[{label}] Calibration")

    run([
        'python', str(script_dir / '06_generate_dataset_plan.py'),
        '--input', str(session_dir),
    ], f"[{label}] Dataset plan")


def next_val(key, current):
    s = PARAM_SCHEDULE[key]
    nv = round(current + s['step'], 6)
    if s['step'] > 0 and nv > s['max']:
        return None
    if s['step'] < 0 and nv < s['max']:
        return None
    return nv


@click.command()
@click.option('--session_dir', required=True)
@click.option('--calibration_dir', default='example/calibration_s21', show_default=True)
@click.option('--output_dir', default=None)
@click.option('--z_start', type=int, default=0, show_default=True)
@click.option('--z_end', type=int, default=300, show_default=True)
@click.option('--z_threshold', type=float, default=10.0, show_default=True)
@click.option('--ref', default='aruco', show_default=True)
@click.option('--max_iter', type=int, default=50, show_default=True,
              help='최대 iteration 수 (초과 시 강제 종료)')
def main(session_dir, calibration_dir, output_dir, z_start, z_end, z_threshold, ref, max_iter):
    session_dir = pathlib.Path(session_dir).absolute()
    output_dir = pathlib.Path(output_dir).absolute() if output_dir else session_dir
    calibration_dir = pathlib.Path(calibration_dir)
    intrinsics = calibration_dir / 's21_intrinsics_1080p.json'
    aruco_config = calibration_dir / 'aruco_config.yaml'
    script_dir = pathlib.Path(ROOT_DIR) / 'droid_slam_s21'
    n_demos = len(list((session_dir / 'demos').glob('demo_*')))

    print(f"Session   : {session_dir}")
    print(f"Z range   : frames {z_start}~{z_end}")
    print(f"Threshold : {z_threshold}mm")
    print(f"Max iter  : {max_iter}")
    print(f"Strategy  : stepwise — filter_thresh → keyframe_thresh → warmup → frontend_window")

    run([
        'python', str(script_dir / '04_detect_aruco.py'),
        '--input_dir', str(session_dir / 'demos'),
        '--camera_intrinsics', str(intrinsics),
        '--aruco_yaml', str(aruco_config),
    ], "04 Detect ArUco (once)")

    cur = {k: PARAM_SCHEDULE[k]['start'] for k in PARAM_SCHEDULE}
    history = []
    iteration = 0
    best_record = None
    param_order = ['filter_thresh', 'keyframe_thresh', 'warmup', 'frontend_window']

    print(f"\n[초기 상태 확인] filter={cur['filter_thresh']}, kf={cur['keyframe_thresh']}, "
          f"warmup={cur['warmup']}, fw={cur['frontend_window']}")
    run_one_iter(session_dir, script_dir, intrinsics, ref,
                 cur['filter_thresh'], cur['keyframe_thresh'],
                 cur['warmup'], cur['frontend_window'], "init")
    init_unstable = count_unstable(session_dir, z_start, z_end, z_threshold)
    init_n = len(init_unstable)
    print(f"  → 초기 Unstable: {init_n} / {n_demos}")
    history.append(dict(label="init", filter=cur['filter_thresh'], kf=cur['keyframe_thresh'],
                        warmup=cur['warmup'], fw=cur['frontend_window'], n_unstable=init_n))
    if best_record is None or init_n < best_record['n']:
        best_record = {'n': init_n, 'params': dict(filter=cur['filter_thresh'], kf=cur['keyframe_thresh'],
                                                    warmup=cur['warmup'], fw=cur['frontend_window'])}

    for param_key in param_order:
        print(f"\n{'='*60}")
        print(f"Tuning: {param_key}  (start={cur[param_key]})")
        print('='*60)

        prev_count = best_record['n']
        best_in_param = {'n': prev_count, 'val': cur[param_key]}

        nv = next_val(param_key, cur[param_key])
        if nv is None:
            print(f"  → {param_key} already at max, skipping")
            continue
        cur[param_key] = nv

        while True:
            if iteration >= max_iter:
                print(f"\n[STOP] max_iter={max_iter} 도달. 탐색 종료.")
                break

            iteration += 1
            label = f"iter{iteration}"
            ft  = cur['filter_thresh']
            kft = cur['keyframe_thresh']
            wu  = cur['warmup']
            fw  = cur['frontend_window']

            print(f"\n[{label}] filter={ft}, kf={kft}, warmup={wu}, fw={fw}")

            run_one_iter(session_dir, script_dir, intrinsics, ref, ft, kft, wu, fw, label)

            unstable = count_unstable(session_dir, z_start, z_end, z_threshold)
            n = len(unstable)
            record = dict(label=label, filter=ft, kf=kft, warmup=wu, fw=fw, n_unstable=n)
            history.append(record)

            if best_record is None or n < best_record['n']:
                best_record = {'n': n, 'params': dict(filter=ft, kf=kft, warmup=wu, fw=fw)}

            if best_in_param is None or n < best_in_param['n']:
                best_in_param = {'n': n, 'val': cur[param_key]}

            print(f"  → Unstable: {n} / {n_demos}")
            for ep, name, md in unstable[:10]:
                print(f"     ep{ep}  max_diff={md:.1f}mm")
            if len(unstable) > 10:
                print(f"     ... ({len(unstable) - 10}개 더)")

            if n == 0:
                print(f"\n[SUCCESS] All stable at [{label}]!")
                break

            improved = (prev_count is None) or (n < prev_count)
            prev_count = n
            nv = next_val(param_key, cur[param_key])

            if not improved or nv is None:
                reason = "no improvement" if not improved else f"{param_key} reached max"
                print(f"  → {reason}, moving to next parameter")
                cur[param_key] = best_in_param['val']
                print(f"  → {param_key} restored to best={cur[param_key]}")
                break

            cur[param_key] = nv

        else:
            break

        if len(count_unstable(session_dir, z_start, z_end, z_threshold)) == 0:
            break

        if iteration >= max_iter:
            break

    print(f"\n{'='*60}")
    print("Tuning history:")
    print(f"  {'iter':<8} {'filter':>8} {'kf':>6} {'warmup':>7} {'fw':>5} {'unstable':>10}")
    print(f"  {'-'*55}")
    for h in history:
        print(f"  {h['label']:<8} {h['filter']:>8.2f} {h['kf']:>6.2f} "
              f"{h['warmup']:>7}  {h['fw']:>5}  {h['n_unstable']:>8}")

    final_unstable = count_unstable(session_dir, z_start, z_end, z_threshold)
    print(f"\nFinal unstable : {len(final_unstable)} / {n_demos}")

    if best_record:
        bp = best_record['params']
        print(f"Best params    : filter={bp['filter']}, kf={bp['kf']}, "
              f"warmup={bp['warmup']}, fw={bp['fw']}  (n={best_record['n']})")

    if len(final_unstable) > 0:
        print(f"\n[불안정 에피소드 목록]")
        for ep, name, md in sorted(final_unstable, key=lambda x: x[2], reverse=True):
            print(f"  ep{ep}  {name}  max_diff={md:.1f}mm")

    suffix = 'success' if len(final_unstable) == 0 else f"best_{best_record['n']}unstable"
    zarr_out = output_dir / f"dataset_{session_dir.name}_{suffix}.zarr.zip"
    run([
        'python', str(script_dir / '07_generate_replay_buffer.py'),
        '-o', str(zarr_out),
        str(session_dir),
    ], "Generate zarr.zip")
    print(f"\nOutput: {zarr_out}")


if __name__ == '__main__':
    main()
