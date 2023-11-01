
import os
import re
import glob
import time
import argparse

import ffmpeg
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

    def __init__(self, videopath, force_fps=None):
        if type(videopath) == str:
            self.videopath = videopath    
            self.cap = cv.VideoCapture(videopath)
            self.fps = self.cap.get(cv.CAP_PROP_FPS)
            self.frame_count = self.get_frame_count(force=True)
            self.duration, _, _, _, _ = self.get_ffmpeg_duration()
            if force_fps:
                force_fps = self.frame_count/self.duration
                pprint("Forced FPS as {}, original value is {}. Leveraging FFMPEG Duration: {}s".format(force_fps, self.fps, self.duration), log_type="WARN")
                self.fps = force_fps
            self.width  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        pprint("FPS:{:.2f}, (Frames: {}, Duration {:.2f} s), \t Video:{} ".format(self.fps, self.frame_count, self.frame_count/self.fps, videopath))

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

    def get_frame_count(self, force=False):
        count = 0
        if not force:
            count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        else:
            with tqdm() as pbar:
                while (self.cap.isOpened()):
                    ret, _ = self.cap.read()
                    if ret:
                        count += 1
                    else:
                        break
                    pbar.update(1)
                else:
                    pprint("OpenCV - loop frame count: {}".format(count))
        return count
   
    def get_ffmpeg_duration(self):
        duration, frame_count, fps_video = 0, 0, 0
        width, height = 0, 0
        _json = ffmpeg.probe(self.videopath)
        if 'format' in _json:
            if 'duration' in _json['format']:
                duration = float(_json['format']['duration'])

        if 'streams' in _json:
            # commonly stream 0 is the video
            for s in _json['streams']:
                if 'duration' in s:
                    duration = float(s['duration'])
                if 'avg_frame_rate' in s and s['codec_type'] == 'video':
                    frame_rate = s['avg_frame_rate'].split('/')
                    fps_video = float(frame_rate[0])
                    width = int(s['width'])
                    height = int(s['height'])
        frame_count = int(duration * fps_video)
        pprint("FFMPEG - duration:{} sec, frames:{}, fps:{}, W:{}, H:{}".format(duration, frame_count, fps_video, width, height))
        return duration, frame_count, fps_video, width, height

    def sample_at_n_fps(self, X, n=1):
        # Determine maxSecs as per original frame rate
        maxFrameNo = X.shape[0] - 1
        maxItems = int((maxFrameNo + 0.5) / (self.fps/n))
        frmItems = np.arange(maxItems + 1)
        indices_fps_round = np.rint(frmItems * (self.fps/n)).astype(int)
        return X[indices_fps_round]
        
    def save_frames_at_n_fps(self, n_fps, output_path):
        filepath, extension = os.path.splitext(self.videopath)
        if output_path is not None:
            basename = os.path.basename(filepath)
            filepath = os.path.join(output_path, basename)
        filepath = "{}-{}".format(filepath, extension[1:])
        os.makedirs(filepath, exist_ok=True)
        # Sample at n_FPS
        indices_nfps = set(self.sample_at_n_fps(np.arange(self.frame_count+1), n_fps))
        print("Sample count >>>", len(indices_nfps))
        frame_id = 0      # Initial frame index
        bad_frames = 0
        # Reset position of the video capture descriptor
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        with tqdm(total=self.frame_count) as pbar:
            while (self.cap.isOpened() and frame_id < self.frame_count):
                frame_id = int(round(self.cap.get(cv.CAP_PROP_POS_FRAMES)))
                ret, frame = self.cap.read()
                if not ret: # Bad frame
                    bad_frames += 1
                    print("{} bad frames - {}/{}".format(bad_frames, frame_id, self.frame_count))
                    if bad_frames >=10: 
                        print("EXIT")
                        break
                elif frame_id in indices_nfps:
                    frame_path = os.path.join(filepath, "frame_{:08d}.jpg".format(frame_id))
                    cv.imwrite(frame_path, frame)
                    pbar.update(int(self.fps))
            else:
                print("frame_id is exhausted")
        pprint('Unreadable frames {}'.format(bad_frames))
        return filepath, indices_nfps

        

"""
Extract Video frames at a specified frame rate
"""
def extract_video_frames(video_infile, n_fps, output_path):
    start = time.time()
    try:
        # TODO: Hard coded force_fps=100.00 to compuute on the fly for AVI files [default=None]
        frmx = FrameExtractor(video_infile, force_fps=100.00)  
        filepath, idx_1fps = frmx.save_frames_at_n_fps(n_fps, output_path)
        pprint('Extracted {}/{} frames to directory {}'.format(len(idx_1fps), frmx.frame_count, filepath))
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
        pprint("FPS:{:.2f}, (Frames: {}, Duration {:.2f}), \t Video:{} ".format(fps_video, frame_count, frame_count/fps_video, video_infile))
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
    python viz.py --video data/VIRAT_S_010204_05_000856_000890.mp4 --save-frames 1
    python viz.py --output-path "./data/output" --save-frames 1 --video data/VIRAT_S_050201_05_000890_000944.mp4 
    python viz.py --output-path "./data/output" --save-frames 1 --video data/cctv_videopipe_2021030114.mp4 
    python viz.py --output-path "./data/output" --save-frames 1 --video-ext ".mp4" --video data/
"""
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Download and process CCTV videos')
    parser.add_argument("--video",           type=str,   default=None)
    parser.add_argument("--video-ext",       type=str,   default="avi",  help="Video extensions: avi (default), mp4, etc")
    parser.add_argument("--save-frames",     type=int,   default=0,  help="0 is default for plotting frames. 1 for sampling at 1FPS and otherwise a number below FPS to sample n`th frame")
    parser.add_argument("--output-path",     type=str,   default=None)
    args=parser.parse_args()
    pprint(args)
    if args.video is not None:           # Process single video sample
        if args.save_frames:
            if os.path.isfile(args.video):
                extract_video_frames(args.video, n_fps=args.save_frames, output_path=args.output_path)
            elif  os.path.isdir(args.video):
                glob_reg = "{}/**/*.{}".format(args.video, args.video_ext)
                for filename in glob.glob(glob_reg, recursive=True):
                    pprint("Processing: {}".format(filename))
                    extract_video_frames(filename, n_fps=args.save_frames, output_path=args.output_path)
        else:    
            # Visualize video 
            plot_video_frames(args.video)
    else:
        pprint("No --video file/folder argument provided: {}".format(args))
    
    pprint("END")
