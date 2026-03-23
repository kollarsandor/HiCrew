#Import OpenCv library
from cv2 import *
import cv2
import numpy as np
import pprint
import sys
import os
import glob
import pyimgsaliency as psal
import matplotlib.cm as cm  
from pathlib import Path

blk_size = 16
def first_img_blk_mean_arr(frame):
    blk_mean_arr = edge_detector(frame)
    return blk_mean_arr

def edge_detector(ref):
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    lowThresh = 0.2*high_thresh
    edges = cv2.Canny(gray,lowThresh,high_thresh)
    
    mat = np.matrix(edges)/255
    h = mat.shape[0]
    w = mat.shape[1]
    blk_mean_arr = []
    for i in range(0,h,blk_size):
        for j in range(0,w,blk_size):
            blk_mean = mat[i:i+blk_size,j:j+blk_size].mean()
            blk_mean_arr.append(blk_mean)
    return blk_mean_arr

def edge_diff(ref_img_blk_mean_arr, curr_img, img_h, img_w):
    curr_img_blk_mean_arr = edge_detector(curr_img)
    diff = [abs(curr_img_blk_mean_arr[t]-ref_img_blk_mean_arr[t]) for t in range(0,len(ref_img_blk_mean_arr))]
    diff2 = [round(val, 1) if val > 0.1 else 0 for val in diff]
    return [round(sum(diff2),2), curr_img_blk_mean_arr]

def cal_edge_diff2(vid):
    file_name = vid + '.mp4'
    cap = cv2.VideoCapture(file_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = int(total_frame/fps)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    scale_h = 128
    scale_w = int(size[0]*1.0*scale_h/size[1])
    vid_path="./sbd_result/"+vid
    if not os.path.exists(vid_path+"/edge_diff.txt"):
        os.mkdir(vid_path)
        os.mkdir(vid_path+"/sbd/")
        os.mkdir(vid_path+"/saliency/")
        f = open(vid_path+"/edge_diff.txt", "w")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flag, frame = cap.read()
        frame = cv2.resize(frame, (scale_w, 128))

        ref_img_blk_mean_arr = first_img_blk_mean_arr(frame)
        for i in range(0, total_frame, fps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            flag, frame = cap.read()
            frame = cv2.resize(frame, (scale_w, 128))
            [diff_val, ref_img_blk_mean_arr] = edge_diff(ref_img_blk_mean_arr, frame, scale_h, scale_w)
            print(int(i/fps), diff_val, total_frame)
            f.write(str(int(i/fps)) + "\t" + str(diff_val) + "\n")
        f.close()

    diff_val_arr = []
    frame_idx_arr = []
    with open(vid_path+"/edge_diff.txt") as ins:
        for line in ins:
            rec =line.replace("\n", "").split("\t")
            frame_idx = int(rec[0])
            diff_val = np.round(float(rec[1]), 0)
            frame_idx_arr.append(frame_idx)

            if diff_val<6:
                diff_val_arr.append(0)
            else:
                diff_val_arr.append(diff_val)
    runs = zero_runs(diff_val_arr)
    shot_start_arr = []
    shot_start = 0
    shot_len = 0
    for t in range(len(runs)): 
        if t == 0:
            shot_start = 0
        else:
            shot_start = runs[t][0]-1
        shot_start_arr.append(shot_start)
        print(shot_start)
    print(diff_val_arr)
    shot_start_arr.append(total_sec)
    print(shot_start_arr)
    
    if not os.path.exists(vid_path + "/shot.txt"):
        cnt = 0
        f = open(vid_path + "/shot.txt", "w")
        for cnt in range(len(shot_start_arr)-1):
            f.write(str(cnt) + "\t" + str(shot_start_arr[cnt]) + "\t" + str(shot_start_arr[cnt+1]-1) + "\t" + str(shot_start_arr[cnt+1]-shot_start_arr[cnt]) + "\n")
            i = int((shot_start_arr[cnt+1]+shot_start_arr[cnt])/2*fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            flag, frame = cap.read()
            frame = cv2.resize(frame, (scale_w, 128))
            cv2.imwrite(vid_path + "/sbd/" + str(cnt) + ".jpg", frame)
        f.close()
    else:
        if not os.path.exists(vid_path + "/SOD/"):
            os.mkdir(vid_path + "/SOD/")


        for cnt in range(len(shot_start_arr)-1):
            if not os.path.exists(vid_path + "/SOD/" + str(cnt) + "_bin.jpg"):

                frame = cv2.imread(vid_path + "/sbd/" + str(cnt) + ".jpg")
                mbd = np.zeros((scale_h,scale_w), np.uint8)
                try:
                    mbd = psal.get_saliency_mbd(frame).astype('uint8')
                except:
                    print("??")
                mbd = cv2.GaussianBlur(mbd,(9,9),0)
                binary_sal = psal.binarise_saliency_map(mbd, method='adaptive')
                colors = cm.jet(np.linspace(0, 1, 255)) # other color schemes: gist_rainbow, nipy_spectral, plasma, inferno, magma, cividis

                blank_image = np.zeros((scale_h,scale_w,3), np.uint8)
                for h in range(scale_h):
                    for w in range(scale_w):
                        blank_image[h][w] = [int(val*255) for val in colors[mbd[h][w]]][0:3][::-1]
                blank_image = cv2.GaussianBlur(blank_image,(5,5),0)

                dst = cv2.addWeighted(frame, 0.5, blank_image, 0.5, 0)
                cv2.imwrite(vid_path + "/SOD/" + str(cnt) + "_frame.jpg", frame)
                cv2.imwrite(vid_path + "/SOD/" + str(cnt) + "_dst.jpg", dst)
                cv2.imwrite(vid_path + "/SOD/" + str(cnt) + "_bin.jpg", 255 * binary_sal.astype('uint8'))


            # added_image = cv2.addWeighted(frame,0.4,255 * mbd.astype('uint8'),0.1,0)
    

    for t in range(0, len(shot_start_arr)-1):
        shot_start = shot_start_arr[t]
        shot_end = shot_start_arr[t+1]
        print(shot_start, shot_end, shot_end-shot_start)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((shot_start+shot_end)/2)*fps)
        flag, frame = cap.read()
        frame = cv2.resize(frame, (scale_w, 128))
        cv2.imwrite( "/Applications/XAMPP/xamppfiles/htdocs/visaug/" + str(shot_start) + "_" + str(shot_end) + ".jpg", frame)
        mbd = psal.get_saliency_mbd(frame).astype('uint8')
        mbd = cv2.GaussianBlur(mbd,(9,9),0)
        img_h, img_w, ch = frame.shape
        blank_image = np.zeros((img_h,img_w,3), np.uint8)
        colors = cm.jet(np.linspace(0, 1, 255)) # other color schemes: gist_rainbow, nipy_spectral, plasma, inferno, magma, cividis
        colors = colors[::-1]
        for h in range(img_h):
            for w in range(img_w):
                blank_image[h][w] = [int(val*255) for val in colors[mbd[h][w]]][0:3][::-1]
        blank_image = cv2.GaussianBlur(blank_image,(5,5),0)

        dst = cv2.addWeighted(frame, 0.5, blank_image, 0.5, 0)
        # added_image = cv2.addWeighted(frame,0.4,255 * mbd.astype('uint8'),0.1,0)
        cv2.imwrite( "/Applications/XAMPP/xamppfiles/htdocs/visaug/"+"/mbd" + str(shot_start) + "_" + str(shot_end) + ".jpg", dst)
        binary_sal = psal.binarise_saliency_map(mbd, method='adaptive')
        

        colors = cm.jet(np.linspace(0, 1, 255)) # other color schemes: gist_rainbow, nipy_spectral, plasma, inferno, magma, cividis
        # colors = colors[::-1]
        print(len(colors*255))
        blank_image = np.zeros((scale_h,scale_w,3), np.uint8)
        for h in range(scale_h):
            for w in range(scale_w):
                blank_image[h][w] = [int(val*255) for val in colors[mbd[h][w]]][0:3][::-1]
        blank_image = cv2.GaussianBlur(blank_image,(5,5),0)

        dst = cv2.addWeighted(frame, 0.5, blank_image, 0.5, 0)

        # added_image = cv2.addWeighted(frame,0.4,255 * mbd.astype('uint8'),0.1,0)
        # cv2.imshow('mbd', mbd)
        # cv2.imshow('blank_image', blank_image)
        # cv2.imshow('added_image', dst)
        # cv2.imshow('binary',255 * binary_sal.astype('uint8'))
        # cv2.imshow("frame", frame)
        # cv2.waitKey(200)

        # cv2.imwrite(vid+"\\sbd\\" + str(shot_start) + "_" + str(shot_end) + ".jpg", frame)
def cal_edge_diff2(vid):
    file_name = vid + '.mp4'
    cap = cv2.VideoCapture(file_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = int(total_frame / fps)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    scale_h = 128
    scale_w = int(size[0] * 1.0 * scale_h / size[1])
    vid_path="./sbd_result/"+vid
    if not os.path.exists(vid_path + "/edge_diff.txt"):
        os.makedirs(vid_path + "/sbd/", exist_ok=True)
        os.makedirs(vid_path + "/saliency/", exist_ok=True)
        f = open(vid_path + "/edge_diff.txt", "w")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flag, frame = cap.read()
        frame = cv2.resize(frame, (scale_w, 128))

        ref_img_blk_mean_arr = first_img_blk_mean_arr(frame)
        for i in range(0, total_frame, fps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            flag, frame = cap.read()
            frame = cv2.resize(frame, (scale_w, 128))
            [diff_val, ref_img_blk_mean_arr] = edge_diff(ref_img_blk_mean_arr, frame, scale_h, scale_w)
            f.write(str(int(i / fps)) + "\t" + str(diff_val) + "\n")
        f.close()

    diff_val_arr = []
    frame_idx_arr = []
    with open(vid_path + "/edge_diff.txt") as ins:
        for line in ins:
            rec = line.strip().split("\t")
            frame_idx = int(rec[0])
            diff_val = np.round(float(rec[1]), 0)
            frame_idx_arr.append(frame_idx)
            diff_val_arr.append(0 if diff_val < 6 else diff_val)

    runs = zero_runs(diff_val_arr)
    shot_start_arr = []
    for t in range(len(runs)):
        if t == 0:
            shot_start = 0
        else:
            shot_start = runs[t][0] - 1
        shot_start_arr.append(shot_start)
    shot_start_arr.append(total_sec)

    print(shot_start_arr)
    return shot_start_arr


def sliding_window(binary_sal):
    img_h, img_w = binary_sal.shape
    mask = np.zeros((img_h, img_w))
    win_size = 16
    for h in range(0,img_h-win_size,win_size):
        for w in range(0,img_w-win_size,win_size):
            # if np.sum(binary_sal[h:h+win_size,w:w+win_size])/255.0>win_size*win_size/16.0:
             if np.sum(binary_sal[h:h+win_size,w:w+win_size])/255.0>win_size*win_size*0.1:
                mask[h:h+win_size,w:w+win_size]=255
    return mask

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges



if __name__=="__main__":
    fd = "./Egoschema_videos/"
    videos=[i.name[:-4] for i in list(Path(fd).iterdir())][:-1]
    print(videos)
    for vid in videos:
    # for vid in [ "qpoRO378qRY", "cdZZpaB2kDM", "7zC8-06198g",  "540vzMlf-54", "LutI8YqJkqM", "KlV0fyDC3Gc"]:

        
        shot_start_arr = cal_edge_diff2(fd + vid)
