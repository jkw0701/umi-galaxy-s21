import sys
import os
import pathlib
import json
import cv2
import numpy as np
import pandas as pd
import click
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

# UMI utils
from umi.common.cv_util import parse_fisheye_intrinsics

@click.command()
@click.option('-v', '--video', required=True, help='Path to raw video file')
@click.option('-c', '--csv', required=True, help='Path to camera_trajectory.csv')
@click.option('-ij', '--intrinsics_json', required=True, help='Path to calibration json')
@click.option('-o', '--output', required=True, help='Path to output video file')
def main(video, csv, intrinsics_json, output):
    video_path = pathlib.Path(video)
    csv_path = pathlib.Path(csv)
    
    # 1. Load Calibration (정확한 카메라 파라미터 로드)
    with open(intrinsics_json, 'r') as f:
        calib_data = json.load(f)
    
    # K matrix 추출 (Unified Camera Model or Pinhole)
    # UMI에서는 보통 GoPro Fisheye를 쓰므로 K 행렬과 왜곡 계수를 가져옵니다.
    # 단순화를 위해 Key 'K'가 있으면 쓰고, 없으면 구성합니다.
    
    # JSON 구조 파악 (일반적인 UMI gopro json 기준)
    if 'K' in calib_data:
        K = np.array(calib_data['K'])
    elif 'camera_matrix' in calib_data: # opencv style
        K = np.array(calib_data['camera_matrix'])
    else:
        # Fallback: UMI example json format
        w, h = calib_data.get('image_width', 1920), calib_data.get('image_height', 1080)
        fx = calib_data.get('fx', w * 0.4) # fisheye is wider -> smaller fx
        fy = calib_data.get('fy', w * 0.4)
        cx = calib_data.get('cx', w / 2)
        cy = calib_data.get('cy', h / 2)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 왜곡 계수 (Distortion Coefficients)
    D = np.zeros(4)
    if 'D' in calib_data:
        D = np.array(calib_data['D'])
    elif 'dist_coeff' in calib_data:
        D = np.array(calib_data['dist_coeff'])

    # 2. Load Trajectory
    # CSV format: frame_idx, timestamp, x, y, z, qx, qy, qz, qw
    try:
        df = pd.read_csv(csv_path)
        traj_data = df.to_numpy() # (N, 9)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    # Frame Index Mapping
    traj_map = {int(row[0]): row for row in traj_data}

    # 3. Process Video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"Processing {total_frames} frames with Camera Matrix:\n{K}")

    # --- 기준점 설정 ---
    # 첫 프레임의 위치를 기준으로 앞에 가상 물체를 배치합니다.
    if 0 not in traj_map and len(traj_map) > 0:
        first_idx = min(traj_map.keys()) # 0번 프레임이 SLAM 실패일 수 있음
    else:
        first_idx = 0
            
    if first_idx in traj_map:
        first_row = traj_map[first_idx]
        pos0 = first_row[2:5]
        rot0 = R.from_quat(first_row[5:9])
        T_w_c0 = np.eye(4)
        T_w_c0[:3, :3] = rot0.as_matrix()
        T_w_c0[:3, 3] = pos0
        
        # 카메라 0번 위치 기준: 앞으로 30cm (z축), 아래로 10cm (y축) 지점에 박스 생성
        # GoPro는 보통 Z축이 정면입니다.
        box_center_c0 = np.array([0, 0.1, 0.3]) 
        box_center_w = (T_w_c0[:3, :3] @ box_center_c0) + T_w_c0[:3, 3]
        
        # 5cm 짜리 박스
        box_size = 0.05
        box_points_w = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    pt = box_center_w + np.array([dx, dy, dz]) * box_size
                    box_points_w.append(pt)
        box_points_w = np.array(box_points_w)
    else:
        print("⚠️ Warning: No trajectory data found for initial frames.")
        box_points_w = None

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in traj_map and box_points_w is not None:
            row = traj_map[frame_idx]
            pos = row[2:5]
            quat = row[5:9] # qx, qy, qz, qw
            
            # 쿼터니언 정규화 (필수)
            norm = np.linalg.norm(quat)
            if norm > 0: quat = quat / norm

            rot = R.from_quat(quat)
            T_w_c = np.eye(4)
            T_w_c[:3, :3] = rot.as_matrix()
            T_w_c[:3, 3] = pos
            
            # W -> C 변환
            T_c_w = np.linalg.inv(T_w_c)
            rvec, _ = cv2.Rodrigues(T_c_w[:3, :3])
            tvec = T_c_w[:3, 3]
            
            # Project Points (Distortion 적용)
            try:
                img_points, _ = cv2.projectPoints(box_points_w, rvec, tvec, K, D)
                img_points = img_points.reshape(-1, 2).astype(int)
                
                # 화면 안에 있는 점만 그림
                valid_points = 0
                for pt in img_points:
                    if 0 <= pt[0] < width and 0 <= pt[1] < height:
                        cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)
                        valid_points += 1
                
                # 라인 그리기 (큐브)
                connections = [
                    (0,1), (1,3), (3,2), (2,0),
                    (4,5), (5,7), (7,6), (6,4),
                    (0,4), (1,5), (2,6), (3,7)
                ]
                for i, j in connections:
                    pt1 = tuple(img_points[i])
                    pt2 = tuple(img_points[j])
                    # 둘 중 하나라도 화면 안에 있으면 그리기
                    if (0 <= pt1[0] < width and 0 <= pt1[1] < height) or \
                       (0 <= pt2[0] < width and 0 <= pt2[1] < height):
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            except Exception as e:
                 # Projection 실패 시 무시
                 pass
            
            status_text = f"Frame: {frame_idx}"
            color = (0, 255, 0)
        else:
            status_text = f"Frame: {frame_idx} (SLAM LOST)"
            color = (0, 0, 255)

        cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx}/{total_frames}", end='\r')
        frame_idx += 1

    cap.release()
    out.release()
    print(f"\n✅ Done! Saved to {output}")

if __name__ == '__main__':
    main()