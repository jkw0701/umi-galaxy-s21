"""
fix_audio_timestamps.py — AAC 오디오 start_pts 오프셋 문제 일괄 수정

Android 앱이 AAC 오디오 타임스탬프를 기기 부팅 시각 기준 절대값으로 기록해서
start_pts 가 수억 단위로 찍히는 문제를 ffmpeg remux로 수정한다.

변경 대상:
  - 입력 폴더 하위 모든 session_*/camera_ultrawide.mp4
  - _fixed / _fixed2 등 이미 수정된 임시 파일은 건너뜀

출력:
  - <input_dir>/fixed/<session_폴더명>/camera_ultrawide.mp4
  - 나머지 비영상 파일(csv, json, jsonl …)은 원본 그대로 복사

Usage:
    python fix_audio_timestamps.py [--input <dir>] [--video <name>] [--dry-run]

    기본값: --input /home/kist/Downloads/robotdatalearning_local
            --video camera_ultrawide.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys


VIDEO_NAME_DEFAULT = "camera_ultrawide.mp4"
OUTPUT_SUBDIR = "fixed"


def get_audio_start_pts(video_path):
    """오디오 스트림의 start_pts 반환. 오디오 없으면 None."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=start_pts,codec_name",
        "-of", "default=noprint_wrappers=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError:
        return None, None

    pts = None
    codec = None
    for line in out.splitlines():
        if line.startswith("start_pts="):
            val = line.split("=", 1)[1].strip()
            try:
                pts = int(val)
            except ValueError:
                pass
        elif line.startswith("codec_name="):
            codec = line.split("=", 1)[1].strip()
    return pts, codec


def fix_timestamps(src_video, dst_video, dry_run=False):
    """
    영상 스트림은 그대로, 오디오 타임스탬프를 0부터 시작하도록 remux.
    -itsoffset 으로 오디오 스트림만 당긴 뒤 두 스트림을 copy 합성.
    """
    pts, codec = get_audio_start_pts(src_video)

    if pts is None:
        # 오디오 스트림 없음 — 영상만 copy
        print(f"  [오디오 없음] copy only: {os.path.basename(src_video)}")
        if not dry_run:
            os.makedirs(os.path.dirname(dst_video), exist_ok=True)
            shutil.copy2(src_video, dst_video)
        return "no_audio"

    # start_pts → 초 단위 오프셋 (AAC timebase = 1/timescale, 일반적으로 1/48000)
    # ffprobe start_time(초) 를 직접 읽는 게 더 안전
    cmd_time = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=start_time",
        "-of", "default=noprint_wrappers=1",
        src_video,
    ]
    try:
        out2 = subprocess.check_output(cmd_time, stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError:
        out2 = ""

    start_time_sec = 0.0
    for line in out2.splitlines():
        if line.startswith("start_time="):
            try:
                start_time_sec = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass

    THRESHOLD = 3600  # 1시간 이상이면 비정상 오프셋으로 판단
    if start_time_sec < THRESHOLD:
        print(f"  [정상] audio start_time={start_time_sec:.2f}s — copy only")
        if not dry_run:
            os.makedirs(os.path.dirname(dst_video), exist_ok=True)
            shutil.copy2(src_video, dst_video)
        return "normal"

    print(f"  [수정 필요] audio start_time={start_time_sec:.0f}s ({start_time_sec/3600:.1f}h) — remux")
    if dry_run:
        return "would_fix"

    os.makedirs(os.path.dirname(dst_video), exist_ok=True)

    # 오디오 스트림만 -itsoffset 으로 당겨서 0부터 시작하게 함
    # Stream #1:v → 영상 원본, Stream #0:a → 오프셋 보정된 오디오
    cmd_fix = [
        "ffmpeg", "-y",
        "-itsoffset", f"-{start_time_sec}",
        "-i", src_video,          # input 0: 오프셋 보정용 (오디오)
        "-i", src_video,          # input 1: 영상 원본
        "-map", "1:v:0",          # 영상 → input 1의 video
        "-map", "0:a:0",          # 오디오 → input 0의 audio (타임스탬프 당겨짐)
        "-c", "copy",
        dst_video,
    ]
    result = subprocess.run(cmd_fix, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ffmpeg 오류:\n{result.stderr[-800:]}")
        return "error"

    # 검증: 수정 후 start_time 확인
    pts2, _ = get_audio_start_pts(dst_video)
    cmd_time2 = cmd_time[:]
    cmd_time2[-1] = dst_video
    try:
        out3 = subprocess.check_output(cmd_time2, stderr=subprocess.DEVNULL, text=True)
        for line in out3.splitlines():
            if line.startswith("start_time="):
                fixed_t = float(line.split("=", 1)[1].strip())
                print(f"    → 수정 후 audio start_time={fixed_t:.3f}s  OK")
    except Exception:
        pass

    return "fixed"


def copy_non_video_files(src_session, dst_session, video_name, dry_run=False):
    """영상 파일 제외 나머지를 그대로 복사."""
    for fname in os.listdir(src_session):
        if fname == video_name or fname.startswith(video_name.replace(".mp4", "")):
            continue  # 영상(원본·임시 fixed 파일) 제외
        src_f = os.path.join(src_session, fname)
        dst_f = os.path.join(dst_session, fname)
        if os.path.isfile(src_f):
            if not dry_run:
                shutil.copy2(src_f, dst_f)


def find_session_dirs(input_dir, video_name):
    """input_dir 아래 video_name 을 포함하는 세션 폴더 목록 반환."""
    sessions = []
    for root, dirs, files in os.walk(input_dir):
        # output 폴더 자체는 건너뜀
        dirs[:] = [d for d in dirs if d != OUTPUT_SUBDIR]
        if video_name in files:
            sessions.append(root)
    return sorted(sessions)


def main():
    parser = argparse.ArgumentParser(description="AAC 오디오 타임스탬프 일괄 수정")
    parser.add_argument("--input", required=True,
                        help="세션 폴더들이 있는 상위 폴더")
    parser.add_argument("--video",   default=VIDEO_NAME_DEFAULT,
                        help="영상 파일명 (기본: camera_ultrawide.mp4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 파일 생성 없이 처리 대상만 출력")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input)
    output_dir = os.path.join(input_dir, OUTPUT_SUBDIR)

    sessions = find_session_dirs(input_dir, args.video)
    if not sessions:
        print(f"처리할 세션 폴더 없음: {input_dir}")
        sys.exit(1)

    print(f"입력 폴더: {input_dir}")
    print(f"출력 폴더: {output_dir}")
    print(f"처리 대상 세션: {len(sessions)}개\n")

    stats = {"fixed": 0, "normal": 0, "no_audio": 0, "error": 0, "would_fix": 0}

    for src_session in sessions:
        # 출력 경로: input_dir 기준 상대 경로를 output_dir 아래에 그대로 재현
        rel = os.path.relpath(src_session, input_dir)
        dst_session = os.path.join(output_dir, rel)

        src_video = os.path.join(src_session, args.video)
        dst_video = os.path.join(dst_session, args.video)

        print(f"[{rel}]")

        if not args.dry_run:
            os.makedirs(dst_session, exist_ok=True)

        status = fix_timestamps(src_video, dst_video, dry_run=args.dry_run)
        stats[status] = stats.get(status, 0) + 1

        if not args.dry_run:
            copy_non_video_files(src_session, dst_session, args.video, dry_run=False)

    print(f"\n{'='*50}")
    print(f"완료! 결과 요약:")
    print(f"  수정됨    : {stats.get('fixed',0)}개")
    print(f"  정상(복사): {stats.get('normal',0)}개")
    print(f"  오디오없음: {stats.get('no_audio',0)}개")
    print(f"  오류      : {stats.get('error',0)}개")
    if args.dry_run:
        print(f"  수정예정  : {stats.get('would_fix',0)}개  (--dry-run 모드)")
    print(f"\n출력 위치: {output_dir}")


if __name__ == "__main__":
    main()
