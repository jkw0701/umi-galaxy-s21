"""
SLAM 마스크 확인 도구 (Galaxy S21)

하드웨어(카메라 마운트, 그리퍼 위치)가 바뀌었을 때 마스크가 그리퍼를
올바르게 덮고 있는지 확인합니다.

마스크가 맞지 않으면 umi/common/cv_util.py의
get_s21_gripper_canonical_polygon() 좌표를 수정하세요.

Usage:
    python droid_slam_s21/visualize_slam_mask.py -i <session_dir>
    python droid_slam_s21/visualize_slam_mask.py -i <session_dir> -o mask_check.png
    python droid_slam_s21/visualize_slam_mask.py -i <session_dir> --frame 60
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import pathlib
import click
import numpy as np
import cv2
import av
import importlib
import umi.common.cv_util as cv_util


def get_frame(video_path: pathlib.Path, frame_idx: int) -> np.ndarray:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                return frame.to_ndarray(format='bgr24')
    raise ValueError(f"Frame {frame_idx} not found in {video_path}")


@click.command()
@click.option('-i', '--input_dir', required=True,
              help='세션 디렉토리 경로 (SLAM 전후 모두 가능)')
@click.option('-o', '--output', default=None,
              help='결과 이미지 저장 경로 (없으면 /tmp/slam_mask_check_s21.png)')
@click.option('--frame', 'frame_idx', type=int, default=30,
              help='확인할 프레임 번호 (기본: 30)')
@click.option('--demo', 'demo_name', default=None,
              help='특정 데모 폴더 이름 (기본: 첫 번째 데모 사용)')
def main(input_dir, output, frame_idx, demo_name):
    # reload so edits to cv_util take effect without restarting
    importlib.reload(cv_util)

    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()

    # 영상 파일 찾기: demos/demo_*/raw_video.mp4 → 없으면 세션 폴더 내 camera_ultrawide.mp4
    if demo_name:
        video_path = input_dir / 'demos' / demo_name / 'raw_video.mp4'
    else:
        candidates = sorted(input_dir.glob('demos/demo_*/raw_video.mp4'))
        if candidates:
            video_path = candidates[0]
            print(f"사용할 데모: {video_path.parent.name}")
        else:
            # SLAM 전 원본 구조: <session>/<timestamp>/camera_ultrawide.mp4
            candidates = sorted(input_dir.rglob('camera_ultrawide.mp4'))
            if not candidates:
                print(f"Error: 영상 파일을 찾을 수 없습니다: {input_dir}")
                print("  찾은 위치: demos/demo_*/raw_video.mp4, **/camera_ultrawide.mp4")
                raise SystemExit(1)
            video_path = candidates[0]
            print(f"사용할 영상 (SLAM 전 원본): {video_path.relative_to(input_dir)}")

    assert video_path.is_file(), f"영상 없음: {video_path}"

    # 여러 프레임 추출 (frame_idx 기준 ±30)
    frame_indices = sorted(set([
        max(0, frame_idx - 30),
        frame_idx,
        frame_idx + 30,
        frame_idx + 60,
    ]))

    panels = []
    for fidx in frame_indices:
        try:
            img = get_frame(video_path, fidx)
        except ValueError:
            continue

        # 마스크를 반투명 핑크로 표시
        overlay = img.copy()
        cv_util.draw_s21_slam_mask(overlay, color=(255, 0, 255))
        blended = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)

        # 프레임 번호 표시
        cv2.putText(blended, f'frame {fidx}', (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 비율 안내선 (20%, 40%, 60%, 80% 높이)
        h, w = blended.shape[:2]
        for pct in [20, 40, 60, 80]:
            y = int(h * pct / 100)
            cv2.line(blended, (0, y), (w, y), (0, 255, 0), 1)
            cv2.putText(blended, f'{pct}%', (5, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        panels.append(blended)

    if not panels:
        print("Error: 프레임을 읽을 수 없습니다.")
        raise SystemExit(1)

    # 2x2 그리드로 합치기
    h, w = panels[0].shape[:2]
    while len(panels) < 4:
        panels.append(np.zeros_like(panels[0]))

    row0 = np.concatenate(panels[:2], axis=1)
    row1 = np.concatenate(panels[2:4], axis=1)
    grid = np.concatenate([row0, row1], axis=0)

    # 안내 텍스트
    cv2.putText(grid, 'PINK = masked (gripper excluded from SLAM)',
                (10, grid.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)

    if output is None:
        output = '/tmp/slam_mask_check_s21.png'

    cv2.imwrite(str(output), grid)
    print(f"저장: {output}")
    print()
    print("마스크가 맞지 않으면 다음 파일의 좌표를 수정하세요:")
    print(f"  {ROOT_DIR}/umi/common/cv_util.py")
    print("  함수: get_s21_gripper_canonical_polygon()")
    print()
    print("수정 후 이 스크립트를 다시 실행해서 확인하세요.")


if __name__ == '__main__':
    main()
