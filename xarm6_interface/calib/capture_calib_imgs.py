#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_calib_imgs.py
---------------------
连续采集 RealSense 标定板图片并保存。
Usage:
  python capture_calib_imgs.py --serial 233622079809 --out_dir calib_imgs

按 <Space> 拍照，<Esc> 退出。
"""

import os, cv2, time, argparse
from pathlib import Path
import numpy as np
import pyrealsense2 as rs

# ───────────── argparse ─────────────
parser = argparse.ArgumentParser()
parser.add_argument('--serial',  required=True, help='317222073552')
parser.add_argument('--out_dir', default='./calib_imgs', help=' /home/xzx/Dro/third_party/xarm6/xarm6_interface/calib/calib_images ')
parser.add_argument('--mode',    choices=['color', 'ir'], default='color',
                    help='采图流：color 或 ir(left)')
parser.add_argument('--fps',     type=int, default=30, help='帧率')
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ───────────── RealSense config ─────────────
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_device(args.serial)

if args.mode == 'color':
    w,h = 1280,720
    cfg.enable_stream(rs.stream.color, w,h, rs.format.bgr8, args.fps)
else:            # left infrared
    w,h = 1280,720
    cfg.enable_stream(rs.stream.infrared, 1, w,h, rs.format.y8, args.fps)

profile = pipe.start(cfg)
align   = rs.align(rs.stream.color) if args.mode=='color' else None
print('[INFO] press <Space> to capture, <Esc> to quit')

# 保存相机内参
if args.mode == 'color':
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
else:
    intr = profile.get_stream(rs.stream.infrared,1).as_video_stream_profile().get_intrinsics()

K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]], dtype=np.float32)
d = np.array(intr.coeffs, dtype=np.float32)
np.save(out_dir/'K.npy', K)
np.save(out_dir/'d.npy', d)
print('[INFO] saved intrinsics to', out_dir)

# ───────────── 采集循环 ─────────────
idx = 0
while True:
    frames = pipe.wait_for_frames()
    if align: frames = align.process(frames)

    if args.mode == 'color':
        frame = frames.get_color_frame()
        img   = np.asanyarray(frame.get_data())          # BGR
    else:
        frame = frames.get_infrared_frame(1)
        img   = np.asanyarray(frame.get_data())          # Gray

    disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim==2 else img.copy()
    cv2.putText(disp, f'{idx:03d}', (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow('CalibCapture', disp)
    k = cv2.waitKey(1) & 0xFF

    if k == 32:                       # Space → 保存
        fname = out_dir / f'img_{idx:03d}.png'
        cv2.imwrite(str(fname), img)
        print(f'[CAPTURE] {fname}')
        idx += 1

    elif k == 27:                     # Esc → 退出
        break

pipe.stop()
cv2.destroyAllWindows()
print('[INFO] Finished, total', idx, 'images saved to', out_dir)

 