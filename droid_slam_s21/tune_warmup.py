"""
warmup 단독 탐색 스크립트.
filter, keyframe_thresh, frontend_window 고정, warmup만 증가하며 탐색.

Usage:
    python droid_slam_s21/tune_warmup.py \
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
import datetime
import click
import numpy as np
import pandas as pd


FIXED_FILTER    = 2.4
FIXED_KF        = 2.8
FIXED_FW        = 25
WARMUP_START    = 26  # --warmup_start 로 override 가능
WARMUP_STEP     = 2
WARMUP_MAX      = 40


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
    slam_failed = []
    unstable = []
    for i, demo_dir in enumerate(demo_dirs):
        z = load_z_mm(demo_dir)
        if z is None:
            slam_failed.append((i, demo_dir.name))
            continue
        seg = z[start:end+1] if len(z) > end else z[start:]
        if len(seg) < 2:
            continue
        max_diff = float(np.abs(np.diff(seg)).max())
        if max_diff > threshold:
            unstable.append((i, demo_dir.name, max_diff))
    return slam_failed, unstable


def run(cmd, desc):
    print(f"\n>>> {desc}")
    print(f"    {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"[ERROR] {desc} failed (returncode={result.returncode})")
        sys.exit(1)


def run_one_iter(session_dir, script_dir, intrinsics, ref, warmup, label):
    run([
        'python', str(script_dir / '03_batch_slam.py'),
        '-i', str(session_dir / 'demos'),
        '--intrinsics', str(intrinsics),
        '--filter_thresh', str(FIXED_FILTER),
        '--keyframe_thresh', str(FIXED_KF),
        '--warmup', str(warmup),
        '--frontend_window', str(FIXED_FW),
        '--overwrite',
    ], f"[{label}] DROID-SLAM  (warmup={warmup})")

    run([
        'python', str(script_dir / '05_run_calibrations_per_episode.py'),
        '--ref', ref,
        str(session_dir),
    ], f"[{label}] Calibration")

    run([
        'python', str(script_dir / '06_generate_dataset_plan.py'),
        '--input', str(session_dir),
    ], f"[{label}] Dataset plan")


@click.command()
@click.option('--session_dir', required=True)
@click.option('--calibration_dir', default='example/calibration_s21', show_default=True)
@click.option('--z_start', type=int, default=0, show_default=True)
@click.option('--z_end', type=int, default=300, show_default=True)
@click.option('--z_threshold', type=float, default=10.0, show_default=True)
@click.option('--ref', default='aruco', show_default=True)
@click.option('--warmup_start', type=int, default=None,
              help='탐색 시작 warmup값 (미지정 시 WARMUP_START 상수 사용)')
def main(session_dir, calibration_dir, z_start, z_end, z_threshold, ref, warmup_start):
    session_dir = pathlib.Path(session_dir).absolute()
    calibration_dir = pathlib.Path(calibration_dir)
    intrinsics = calibration_dir / 's21_intrinsics_1080p.json'
    aruco_config = calibration_dir / 'aruco_config.yaml'
    script_dir = pathlib.Path(ROOT_DIR) / 'droid_slam_s21'

    if not (session_dir / 'demos').exists():
        print("\n[준비] demos 폴더 없음 → 00_process_videos.py 실행")
        run([
            'python', str(script_dir / '00_process_videos.py'),
            str(session_dir),
        ], "00 Process Videos")

    run([
        'python', str(script_dir / '04_detect_aruco.py'),
        '--input_dir', str(session_dir / 'demos'),
        '--camera_intrinsics', str(intrinsics),
        '--aruco_yaml', str(aruco_config),
    ], "04 Detect ArUco (once)")

    n_demos = len(list((session_dir / 'demos').glob('demo_*')))

    start_wu = warmup_start if warmup_start is not None else WARMUP_START

    print(f"Session   : {session_dir}")
    print(f"Fixed     : filter={FIXED_FILTER}, kf={FIXED_KF}, fw={FIXED_FW}")
    print(f"Warmup    : {start_wu} → +{WARMUP_STEP} → max {WARMUP_MAX}")
    print(f"Threshold : {z_threshold}mm  |  Z range: {z_start}~{z_end}")

    log_path = session_dir / 'tune_warmup_log.json'
    if log_path.exists():
        log_data = json.load(open(log_path))
    else:
        log_data = {
            'session_dir': str(session_dir),
            'fixed': {'filter': FIXED_FILTER, 'kf': FIXED_KF, 'fw': FIXED_FW},
            'threshold_mm': z_threshold,
            'z_range': [z_start, z_end],
            'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'iterations': [],
        }

    history = []
    best_record = None
    warmup = start_wu

    while warmup <= WARMUP_MAX:
        label = f"warmup{warmup}"
        print(f"\n{'='*60}")
        print(f"[{label}] filter={FIXED_FILTER}, kf={FIXED_KF}, warmup={warmup}, fw={FIXED_FW}")
        print('='*60)

        run_one_iter(session_dir, script_dir, intrinsics, ref, warmup, label)

        slam_failed, unstable = count_unstable(session_dir, z_start, z_end, z_threshold)
        n_failed = len(slam_failed)
        n_unstable = len(unstable)
        n_problem = n_failed + n_unstable
        history.append(dict(warmup=warmup, n_slam_failed=n_failed, n_unstable=n_unstable, n_problem=n_problem))

        print(f"  → SLAM 실패: {n_failed} / {n_demos}")
        for ep, name in slam_failed:
            print(f"     ep{ep:02d}  {name}")
        print(f"  → Unstable : {n_unstable} / {n_demos}")
        for ep, name, md in unstable:
            print(f"     ep{ep:02d}  max_diff={md:.1f}mm")
        print(f"  → 총 문제  : {n_problem} / {n_demos}")

        iter_record = {
            'warmup': warmup,
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'n_slam_failed': n_failed,
            'n_unstable': n_unstable,
            'n_problem': n_problem,
            'slam_failed': [{'ep': ep, 'name': name} for ep, name in slam_failed],
            'unstable': [{'ep': ep, 'name': name, 'max_diff_mm': round(md, 2)} for ep, name, md in unstable],
        }
        log_data['iterations'] = [r for r in log_data['iterations'] if r['warmup'] != warmup]
        log_data['iterations'].append(iter_record)
        log_data['iterations'].sort(key=lambda r: r['warmup'])
        log_data['last_updated'] = datetime.datetime.now().isoformat(timespec='seconds')
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"  → 로그 저장: {log_path}")

        if best_record is None or n_problem < best_record['n']:
            best_record = {'n': n_problem, 'warmup': warmup}

        if n_problem == 0:
            print(f"\n[SUCCESS] All stable at warmup={warmup}!")
            break

        warmup += WARMUP_STEP

    print(f"\n{'='*60}")
    print("Tuning history:")
    print(f"  {'warmup':>8} {'slam_fail':>10} {'unstable':>10} {'problem':>10}")
    print(f"  {'-'*42}")
    for h in history:
        print(f"  {h['warmup']:>8}  {h['n_slam_failed']:>9}  {h['n_unstable']:>9}  {h['n_problem']:>9}")

    if best_record:
        print(f"\nBest warmup : {best_record['warmup']}  (problem={best_record['n']})")
        print(f"Best params : filter={FIXED_FILTER}, kf={FIXED_KF}, "
              f"warmup={best_record['warmup']}, fw={FIXED_FW}")


if __name__ == '__main__':
    main()
