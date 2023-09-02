# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import argparse
import numpy as np
import numpy.random as random
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/car.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cotracker_stride_4_wind_12.pth",
        help="cotracker model",
    )
    parser.add_argument("--grid_size", type=int,
                        default=30, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame ",
    )

    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )

    args = parser.parse_args()

    # cat multi photo as a video
    video = []
    # photo_folder = '/kube/home/DATASET/processed/guangwei/20230724/停车场_朝阳温榆河公园14号停车场/20230724141332/CAMERA/FW/'
    photo_folder = '/kube/home/DATASET/processed/guangwei/20230724/停车场_朝阳温榆河公园2号停车场/20230724163722_passby/CAMERA/FW'
    save_dir = "./saved_videos/停车场_朝阳温榆河公园2号停车场"
    intervel = 1
    frame = 100
    filename = f'1920_wenyuhe2_{frame}frame_{intervel}intervel_noresize_win12'
    resize = [1920,1080]
    photo_sequnce_tmp = sorted(os.listdir(photo_folder))[:frame]
    photo_sequnce = photo_sequnce_tmp[::intervel]
    for i in range(len(photo_sequnce)):
        crop_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(photo_folder, photo_sequnce[i])), cv2.COLOR_BGR2RGB), resize, interpolation=cv2.INTER_LINEAR)
        video.append(torch.tensor(crop_img, dtype=torch.float)[None])

    video = torch.stack(video, 0).permute(
        1, 0, 4, 2, 3)  # T B H W C -> B T C H W

    # # load the input video frame by frame
    # video = read_video_from_path(args.video_path)
    # video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()   # T H W C -> B T C H W
    # segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    # segm_mask = torch.from_numpy(segm_mask)[None, None]

    model = CoTrackerPredictor(checkpoint=args.checkpoint)
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)

    # queries_tmp = torch.tensor([
    #     [0., 535., 800.-10*i] for i in range(20)    
    # ])

    queries = torch.tensor([
    [0., 156., 739.],
    [0., 156., 736.],
    [0., 156., 730.],

    [0., 380., 696.], # borad
    [0., 385., 696.],
    [0., 400., 656.],
    [0., 385., 627.], 
    [0., 390., 627.], 

    [10., 360., 670.], # borad
    [10., 365., 670.],
    [10., 380., 660.],
    [10., 365., 630.], 
    [10., 370., 630.], 
    
    [0., 425., 705.],
    [0., 420., 705.],
    [0., 430., 630.],
    [0., 425., 630.],

    [0., 780., 810.], # arro
    [0., 780., 800.],
    [0., 780., 795.],
    [0., 780., 790.],

    [0., 775., 793.],

    [0., 772., 793.],
    [0., 740., 793.],

    [0., 770., 690.],
    [0., 770., 710.],

    [0., 1700., 600.],
    [0., 1700., 770.],
    [0., 1480., 625.],
    [0., 1480., 770.]
]
    )

    # queries = torch.cat((queries,queries_tmp),0)
    if torch.cuda.is_available():
        queries = queries.cuda()

    # visual points
    size, dpi, n = 8, 200, 2    
    img0 = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(photo_folder, photo_sequnce[0])), cv2.COLOR_BGR2RGB), resize, interpolation=cv2.INTER_LINEAR)
    
    xaxis = np.linspace(0, video.shape[4], 20)
    yaxis = np.linspace(0, video.shape[3], 20)

    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.imshow(img0, vmin=0, vmax=255)
    ax.get_xaxis().set_ticks(xaxis)
    ax.get_yaxis().set_ticks(yaxis)
    for j in range(len(queries)):
        ax.scatter(queries[j][1].item(), queries[j][2].item(), c='r', s=.5)

    plt.tight_layout(pad=.5)
    plt.savefig(str('test.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Save img to {}".format(str('test.png')))

    pred_tracks, pred_visibility = model(
        video,
        queries=queries[None],
        # grid_size=args.grid_size,
        # grid_query_frame=args.grid_query_frame,
        # backward_tracking=args.backward_tracking,
        # segm_mask=segm_mask
    )
    # pred_tracks, pred_visibility = torch.load('track.nii.gz'), torch.load('visibility.nii.gz')
    print("computed")

    # save a video with predicted tracks
    # seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir=save_dir, mode='cool',tracks_leave_trace=0,linewidth=1.5, fps=10)
    vis.visualize(video, pred_tracks, pred_visibility,
                  query_frame=args.grid_query_frame, filename=filename)

    print('ok')
