"""
DROID-SLAM Z축 불안정 에피소드 식별 스크립트.

특정 프레임 구간(기본 90~140)에서 연속 프레임 간 Z 변화량(frame-to-frame diff)을
기준으로 불안정한 에피소드를 찾아 출력합니다.
불안정 에피소드에 대해 원본 세션 폴더명(session_*)도 함께 출력합니다.

Usage:
    python droid_slam_s21/check_slam_quality.py -i <session_dir>
    python droid_slam_s21/check_slam_quality.py -i <session_dir> --start 90 --end 140 --threshold 5.0
    python droid_slam_s21/check_slam_quality.py -i <session_dir> -o slam_quality.png
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import pathlib
import json
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    z_mm = df_valid['z'].to_numpy() / scale * 1000
    return z_mm


def build_timestamp_map(session_dir: pathlib.Path) -> dict:
    """start_timestamp_ns → 원본 세션 폴더명(session_*) 매핑 딕셔너리 반환."""
    ts_map = {}
    for orig_dir in session_dir.iterdir():
        if not orig_dir.is_dir():
            continue
        meta_path = orig_dir / 'metadata.json'
        if not meta_path.exists():
            continue
        try:
            meta = json.load(open(meta_path))
            ts = meta.get('start_timestamp_ns')
            if ts is not None:
                ts_map[int(ts)] = orig_dir.name
        except Exception:
            pass
    return ts_map


@click.command()
@click.option('-i', '--input', 'session_dir', required=True, help='session directory (contains demos/)')
@click.option('--start', type=int, default=90, show_default=True, help='분석 시작 프레임')
@click.option('--end',   type=int, default=140, show_default=True, help='분석 종료 프레임')
@click.option('--threshold', type=float, default=5.0, show_default=True,
              help='불안정 판단 기준: 연속 프레임 간 Z 최대 변화량 (mm)')
@click.option('-o', '--output', default=None, help='그래프 저장 경로 (없으면 화면 출력)')
def main(session_dir, start, end, threshold, output):
    session_dir = pathlib.Path(session_dir)
    demos_dir = session_dir / 'demos'

    ts_map = build_timestamp_map(session_dir)

    demo_dirs = sorted([d for d in demos_dir.iterdir()
                        if d.is_dir() and d.name.startswith('demo_')])

    results = []
    for i, demo_dir in enumerate(demo_dirs):
        z = load_z_mm(demo_dir)

        orig_name = '?'
        meta_path = demo_dir / 'metadata.json'
        if meta_path.exists():
            try:
                meta = json.load(open(meta_path))
                ts = meta.get('start_timestamp_ns')
                if ts is not None:
                    orig_name = ts_map.get(int(ts), '?')
            except Exception:
                pass

        if z is None:
            print(f"  [ep {i:>3}] camera_trajectory.csv 없음, 건너뜀")
            continue

        seg = z[start:end+1] if len(z) > end else z[start:]
        if len(seg) < 2:
            results.append({'ep': i, 'name': demo_dir.name, 'orig_name': orig_name,
                            'max_diff': 0.0, 'mean_diff': 0.0, 'n_frames': len(z), 'z': z})
            continue

        diffs = np.abs(np.diff(seg))
        results.append({
            'ep': i,
            'name': demo_dir.name,
            'orig_name': orig_name,
            'max_diff': float(diffs.max()),
            'mean_diff': float(diffs.mean()),
            'n_frames': len(z),
            'z': z,
        })

    unstable = [r for r in results if r['max_diff'] > threshold]
    stable   = [r for r in results if r['max_diff'] <= threshold]

    print(f"\n분석 구간: 프레임 {start}~{end}  |  임계값: {threshold} mm/frame")
    print(f"총 {len(results)}개 에피소드  →  불안정: {len(unstable)}개  /  안정: {len(stable)}개")
    print()

    results_sorted = sorted(results, key=lambda x: x['max_diff'], reverse=True)
    print(f"{'Ep':>3} | {'max_diff(mm)':>12} | {'mean_diff(mm)':>13} | {'n_frames':>8} | {'원본 폴더명':<30} | 상태")
    print('-' * 90)
    for r in results_sorted:
        flag = 'UNSTABLE' if r['max_diff'] > threshold else 'stable'
        print(f"{r['ep']:>3} | {r['max_diff']:>12.2f} | {r['mean_diff']:>13.2f} | "
              f"{r['n_frames']:>8} | {r['orig_name']:<30} | {flag}")

    print()
    print("불안정 에피소드 원본 폴더명:")
    for r in sorted(unstable, key=lambda x: x['orig_name']):
        print(f"  {r['orig_name']}  (ep{r['ep']}, max_diff={r['max_diff']:.1f}mm)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'SLAM Z Quality  |  Frames {start}~{end}  |  Threshold: {threshold}mm/frame',
                 fontsize=12)

    ax1 = axes[0]
    for r in stable:
        ax1.plot(r['z'], color='#90CAF9', lw=0.7, alpha=0.5)
    for r in unstable:
        ax1.plot(r['z'], color='#F44336', lw=1.0, alpha=0.8, label=f"ep{r['ep']}({r['orig_name']})")

    ax1.axvspan(start, end, color='yellow', alpha=0.15, label=f'Analysis range ({start}~{end})')
    ax1.axhline(0, color='gray', lw=0.5, ls='--')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Z (mm)')
    ax1.set_title(f'Z trajectory  (blue=stable {len(stable)}, red=unstable {len(unstable)})')
    ax1.grid(True, alpha=0.3)
    if len(unstable) <= 15:
        ax1.legend(fontsize=7, loc='upper right')

    ax2 = axes[1]
    eps    = [r['ep'] for r in results]
    maxds  = [r['max_diff'] for r in results]
    colors = ['#F44336' if m > threshold else '#90CAF9' for m in maxds]
    ax2.bar(eps, maxds, color=colors, edgecolor='white', linewidth=0.5)
    ax2.axhline(threshold, color='black', ls='--', lw=1.2, label=f'Threshold {threshold}mm')
    ax2.set_xlabel('Episode index')
    ax2.set_ylabel('max frame-to-frame Z diff (mm)')
    ax2.set_title('Max Z diff per episode')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"\n그래프 저장: {output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
