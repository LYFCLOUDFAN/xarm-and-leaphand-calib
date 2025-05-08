#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realsense_intrinsics_calib.py
使用 ArUco-Chessboard 标定 RealSense 左红外相机内参。
输出与旧脚本同名：K.npy / d.npy / init_X_BaseleftCamera.npy
运行环境：xarm 或 foundationpose 的任一 Conda，要求 OpenCV>=4.7
-------------------------------------------------------------------
准备：
1. 把多张标定板照片放到 --img_dir (默认 ./calib_imgs) 目录
   例如:
       calib_imgs/
           img_000.jpg
           img_001.jpg
           ...
   也可以只放你给的 /xarm6_interface/calib/aruco_calib.jpg
2. 板型：Aruco + Chessboard 5×7 (7 行、5 列方格，格长可实测填入)
-------------------------------------------------------------------
"""

import argparse, os, glob, pickle, time, json
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs

# ---------- 参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--serial', default='317222073552',
                    help='RealSense 左红外相机序列号')
parser.add_argument('--img_dir', default='./calib_imgs',
                    help='存放标定板照片的文件夹')
parser.add_argument('--square', type=float, default=0.02,   # ← 方格实测边长(米)
                    help='棋盘格单元边长 (m)')
parser.add_argument('--aruco',  type=str,   default='DICT_4X4_250',
                    help='Aruco 字典，OpenCV 支持的名字')
args = parser.parse_args()

# ---------- ArUco 棋盘参数 ----------
rows, cols = 10, 10             # chessboard: 行(垂直) × 列(水平)
aruco_dict = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, args.aruco))
board = cv2.aruco.CharucoBoard((cols, rows),
                               squareLength=args.square,
                               markerLength=args.square*0.75,
                               dictionary=aruco_dict)

# ---------- 收集角点 ----------
img_paths = sorted(glob.glob(str(Path(args.img_dir)/'*')))
assert img_paths, f'文件夹 {args.img_dir} 里没有图像！'

all_corners, all_ids = [], []
img_size = None
for p in img_paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'[WARN] 读不到 {p}')
        continue
    img_size = img.shape[::-1]   # (w,h)
    corners, ids, _ = cv2.aruco.detectMarkers(
        img, aruco_dict)
    if ids is None or len(ids) < 4:
        print(f'[WARN] {p} 角点不足')
        continue
    _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, img, board)
    if ch_corners is not None and ch_ids.size:
        all_corners.append(ch_corners)
        all_ids.append(ch_ids)
    print(f'[INFO] {p}: 检到 {len(corners)} ArUco，{0 if ch_ids is None else len(ch_ids)} Charuco')

assert len(all_corners) >= 3, '有效图片不足 3 张，无法稳定标定！'

# ---------- 标定 ----------
ret, K, d, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=img_size,
    cameraMatrix=None,
    distCoeffs=None)

print('[INFO] 标定 RMS reprojection error =', ret)
print('[INFO] Camera K =\n', K)
print('[INFO] Dist coeffs =', d.ravel())

# ---------- 保存 ----------
save_dir = Path('/home/xzx/Dro/third_party/xarm6/data/camera/317222073552')
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir/'K.npy', K.astype(np.float32))
np.save(save_dir/'d.npy', d.astype(np.float32))

# 若你已有手动测得的左右红外外参，可直接留用
# 这里把之前 script 里的 init_X_BaseleftCamera.npy 原样复制
old_init = Path(save_dir/'init_X_BaseleftCamera.npy')
if not old_init.exists():
    np.save(old_init, np.eye(4, dtype=np.float32))

print(f'[INFO] 已写入 {save_dir}/K.npy, d.npy, init_X_BaseleftCamera.npy')

 
