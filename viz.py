
import os
import re
import glob
import time
import argparse

import cv2 as cv
import numpy as np
from tqdm import tqdm

import detect 

""" 
    Print the log with timestamp 
"""
def pprint(log_text, log_type="INFO", log_name="VIZ"):
    print("[{}] [{}][{}] - {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), log_name, log_type, log_text))



# ------------- OpenCV/Visualize ---------------- #


class FrameExtractor:

    def __init__(self, videopath):
        if type(videopath) == str:
            self.videopath = videopath    
            self.cap = cv.VideoCapture(videopath)
            self.fps = self.cap.get(cv.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        pprint("FPS:{:.2f}, (Frames: {}), \t Input Video:{} ".format(self.fps, self.frame_count, videopath))

    def vcrsecs_to_framenumber(self, vcrsecs):
        frame_no = int(np.rint(vcrsecs * self.fps))
        if frame_no >= self.frame_count(): # Frame OOB
            frame_no = None
        return frame_no

    # TODO: Slows down after a while when dealing with 100K frames
    def image_from_frame(self, framenumber):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, framenumber)
        _, img = self.cap.read()
        return img

    def frame_count(self):
        return self.frame_count
   
    def sample_at_n_fps(self, X, n=1):
        # Determine maxSecs as per original frame rate
        maxFrameNo = X.shape[0] - 1
        maxItems = int((maxFrameNo + 0.5) / (self.fps/n))
        frmItems = np.arange(maxItems + 1)
        indices_fps_round = np.rint(frmItems * (self.fps/n)).astype(int)
        return X[indices_fps_round]
        
    def save_frames_at_n_fps(self, n_fps):
        filepath, extension = os.path.splitext(self.videopath)
        #if output_dir is None:
        os.makedirs(filepath, exist_ok=True)
        # Sample 
        indices_nfps = set(self.sample_at_n_fps(np.arange(self.frame_count+1), n_fps))
        frame_id = 0      # Initial frame index
        bad_frames = 0
        with tqdm(total=self.frame_count) as pbar:
            while (self.cap.isOpened() and frame_id < self.frame_count):
                frame_id = int(round(self.cap.get(cv.CAP_PROP_POS_FRAMES)))
                ret, frame = self.cap.read()
                pbar.update(1)
                if not ret: # Bad frame
                    bad_frames += 1
                    continue
                elif frame_id in indices_nfps:
                    frame_path = os.path.join(filepath, "frame_{:08d}.jpg".format(frame_id))
                    cv.imwrite(frame_path, frame)
            else:
                print("frame_id is exhausted")
        pprint('Extracted {}/{} frames to directory {}. Unreadable frames {}'.format(len(indices_nfps), self.frame_count, filepath, bad_frames))
        return indices_nfps

        

"""
Extract Video frames at a specified frame rate
"""
def extract_video_frames(video_infile, n_fps):
    start = time.time()
    try:
        frmx = FrameExtractor(video_infile)
        idx_1fps = frmx.save_frames_at_n_fps(n_fps)
        
    except KeyboardInterrupt:
        pprint('Interrupted ctrl-c')
    finally:
        if frmx.cap:
            frmx.cap.release()
        pprint("Completed in {:.3f} Sec - {} ".format(time.time()-start, video_infile))
    return


"""
Visualize the frame with control/progress bar
"""
def plot_video_frames(video_infile, do_visualize=True):
    start = time.time()
    try:
        frmx = FrameExtractor(video_infile)
        frame_count = frmx.frame_count
        fps_video = frmx.fps
        pprint("FPS:{:.2f}, (Frames: {}), \t Video:{} ".format(fps_video, frame_count, video_infile))
        good_frames = 0
        bad_frames = 0

        cv_window_name = "FPS:{}, Frames:{}, Video:{}".format(fps_video, frame_count, os.path.basename(video_infile))
        def onCurrentFrameTrackbarChange(trackbarValue):
            pprint("Current Frames Value: {}".format(trackbarValue))
            pass
        if do_visualize:
            cv.namedWindow(cv_window_name) 
            cv.createTrackbar('current-frame', cv_window_name, 1, frame_count, onCurrentFrameTrackbarChange)

        frame_id = 0      # Initial frame index
        while frame_id < frame_count:
            img_from_frame = frmx.image_from_frame(frame_id)
            if img_from_frame is None: # Bad frame, skip
                bad_frames += 1
                continue

            if do_visualize:
                cv.setTrackbarPos('current-frame', cv_window_name, frame_id)
                detect.detect(img_from_frame, do_overlay=True)
                cv.imshow(cv_window_name, img_from_frame)

                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    pprint("Quit")
                    break
                
                frame_id = cv.getTrackbarPos('current-frame', cv_window_name)
                frame_id = frame_id + 1
        else:
            print("frame_id is exhausted")
    except KeyboardInterrupt:
        pprint('Interrupted ctrl-c')
    finally:
        # The following frees up resources and closes all windows
        if frmx.cap:
            frmx.cap.release()

        cv.destroyAllWindows()
        pprint("Completed in {} Sec \t - {} ".format(time.time()-start, video_infile))

    pprint('Video {} has {} bad frames.'.format(video_infile, bad_frames))
    return

"""
Usage: 
    Download and extract frames from the video for assessment
    Check the correctness of Metadata

    python viz.py --video data/VIRAT_S_050201_05_000890_000944.mp4
    python viz.py --video data/VIRAT_S_010204_05_000856_000890.mp4 --save-frames True
    python viz.py --video data/VIRAT_S_050201_05_000890_000944.mp4 --save-frames True
    python viz.py --video data/cctv_videopipe_2021030114.mp4 --save-frames True
"""
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Download and process CCTV videos')
    parser.add_argument("--video",           type=str,   default="")
    parser.add_argument("--save-frames",     type=str,   default=None,  help="Skip frames is 0 to sample at 1FPS and otherwise a number below FPS to sample n`th frame")
    args=parser.parse_args()

    if args.video is not None:           # Process single video sample
        if args.save_frames is not None:
            extract_video_frames(args.video, n_fps=1)
        else:    
            # Visualize video 
            plot_video_frames(args.video)
    else:
        pprint("No --video-file argument provided: {}".format(args))
    
    pprint("END")
