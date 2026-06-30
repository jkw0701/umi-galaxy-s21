"""
Demo video reviewer — browse collected UMI demo videos for selection.

Usage:
    python review_demos.py <demos_dir> [start_index]

Keys:
    n     : next video
    p     : previous video
    q     : quit
    +/=   : increase speed (0.25x step)
    -     : decrease speed (0.25x step)
    space : pause / resume
"""

import cv2
import pathlib
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python review_demos.py <demos_dir> [start_index]")
        sys.exit(1)

    video_dir = pathlib.Path(sys.argv[1]).expanduser().resolve()
    if not video_dir.is_dir():
        print(f"Directory not found: {video_dir}")
        sys.exit(1)

    videos = sorted(
        p for p in video_dir.rglob("*.mp4")
        if "demos" not in p.parts
    )
    if not videos:
        print("No video files found.")
        sys.exit(1)

    print(f"Found {len(videos)} videos in {video_dir}")
    print("n: next  |  p: prev  |  q: quit  |  +/-: speed  |  space: pause/resume")

    idx = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
    idx = max(0, min(idx, len(videos) - 1))
    speed = 1.0

    while True:
        path = videos[idx]
        title = f"[{idx+1}/{len(videos)}] {path.parent.name}/{path.name}"
        print(f"\nPlaying: {title}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  Failed to open: {path}")
            idx = (idx + 1) % len(videos)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        action = None
        paused = False

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    action = 'n'
                    break
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (w * 2 // 3, h * 2 // 3))

            pause_mark = "  [PAUSED]" if paused else ""
            cv2.imshow(f"{title}  [{speed:.2f}x]{pause_mark}", frame)
            delay = 50 if paused else max(1, int(1000 / fps / speed))
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                action = 'q'
                break
            elif key == ord('n'):
                action = 'n'
                break
            elif key == ord('p'):
                action = 'p'
                break
            elif key == ord(' '):
                paused = not paused
                print(f"  {'Paused' if paused else 'Resumed'}")
            elif key in (ord('+'), ord('=')):
                speed = round(min(speed + 0.25, 8.0), 2)
                print(f"  Speed: {speed}x")
            elif key == ord('-'):
                speed = round(max(speed - 0.25, 0.25), 2)
                print(f"  Speed: {speed}x")

        cap.release()
        cv2.destroyAllWindows()

        if action == 'q':
            print("Quit.")
            break
        elif action == 'p':
            idx = (idx - 1) % len(videos)
        else:
            idx = (idx + 1) % len(videos)


if __name__ == '__main__':
    main()
