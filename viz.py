
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
            self.cap = cv.VideoCapture(videopath)
            self.FPS = self.cap.get(cv.CAP_PROP_FPS)
            self.FrameCount = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        elif type(videopath) == dict:
            self.FPS = videopath['FPS']
            self.FrameCount = videopath['FrameCount']
    def vcrsecs_to_framenumber(self, vcrsecs):
        frame_no = int(np.rint(vcrsecs * self.FPS))
        if frame_no >= self.frame_count(): # Frame OOB
            frame_no = None
        return frame_no
    # Method valid only when contructor argument is a video file path
    def image_from_frame(self, framenumber):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, framenumber)
        _, img = self.cap.read()
        return img
    def frame_count(self):
        return self.FrameCount
    # Given X an array sampled at full frame rate self.FPS, return array sampled at 1 fps
    def sample_at_1fps(self, X):
        # Determine maxSecs
        maxFrameNo = X.shape[0] - 1
        maxSecs = int((maxFrameNo + 0.5) / self.FPS)
        secs = np.arange(maxSecs + 1)
        indices_fps_round = np.rint(secs * self.FPS).astype(int)
        return X[indices_fps_round]

"""
Visualize the frame with control/progress bar
"""
def plot_video_frames(video_infile, do_visualize=True):
    start = time.time()
    try:
        xtor = FrameExtractor(video_infile)
        frame_count = xtor.FrameCount
        fps_video = xtor.FPS
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
            img_from_frame = xtor.image_from_frame(frame_id)
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
        if xtor.cap:
            xtor.cap.release()

        cv.destroyAllWindows()
        pprint("Completed in {} Sec \t - {} ".format(time.time()-start, video_infile))

    pprint('Video {} has {} bad frames.'.format(video_infile, bad_frames))
    return

"""
Usage: 
    Download and extract frames from the video for assessment
    Check the correctness of Metadata

    python viz.py --video data/VIRAT_S_010204_05_000856_000890.mp4
    python viz.py --video data/VIRAT_S_050201_05_000890_000944.mp4

    python viz.py --video "/home/rahul/workspace/data/nbcuni/ride-safety/Shared - Hitachi/Fly 04 4.4/Fly 04 4.4/usf simpsons south fly 04 4.4 - 0900 to 1200.avi"
    python viz.py --video "/home/rahul/workspace/data/nbcuni/ride-safety/Shared - Hitachi/Fly 08 5.4/Fly 08 5.4/usf simpsons south fly 08 5.4 - 0900 to 1200.avi"

"""
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Download and process CCTV videos')
    parser.add_argument("--video",           type=str,   default="")
    parser.add_argument("--output-dir",      type=str,   default="./", help="Output directory where we expect to write intermediate and results")
    parser.add_argument("--skip-frames",     type=int,   default=None,  help="Skip frames is 0 to sample at 1FPS and otherwise a number below FPS to sample n`th frame")
    args=parser.parse_args()

    output_dir      = args.output_dir if args.output_dir is not None else "./"
    if args.video is not None:           # Process single video sample
        # Visualize video 
        plot_video_frames(os.path.join(args.output_dir, args.video))
    else:
        pprint("No --video-file argument provided: {}".format(args))
    
    pprint("END")
