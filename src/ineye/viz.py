
import os
import glob
import time
import csv, json
import argparse
import traceback

import subprocess

import ffmpeg
import cv2 as cv
import numpy as np
from tqdm import tqdm

""" 
    Print the log with timestamp 
"""
def pprint(log_text, log_type="INFO", log_name="VIZ"):
    print("[{}] [{}][{}] - {}".format(time.strftime("%Y-%m-%dT%H:%M:%S"), log_name, log_type, log_text))



class FrameExtractor:
    """
    OpenCV & FFMPEG utility to sample frames in a video 
    """
    def __init__(self, videopath, force_fps=True):
        if type(videopath) == str:
            self.videopath = videopath    
            self.cap = cv.VideoCapture(videopath)
            self.fps = self.cap.get(cv.CAP_PROP_FPS)
            self.frame_count = self.get_frame_count(force=True)
            self.duration, _, _, _, _ = self.get_ffmpeg_duration()
            if force_fps:
                # TODO: Hard coded force_fps=True to compute on the fly for AVI files [default=None]
                force_fps = self.frame_count/self.duration
                pprint("Forced FPS as {:.4f}, original value is {:.2f}. Leveraging FFMPEG Duration: {}s".format(force_fps, self.fps, self.duration), log_type="WARN")
                self.fps = force_fps
            self.width  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        pprint("FPS:{:.2f}, (Frames: {}, Duration {:.2f} s), \t Video:{} ".format(self.fps, self.frame_count, self.frame_count/self.fps, videopath))

    def vcrsecs_to_frame_id(self, vcrsecs):
        frame_id = int(np.rint(vcrsecs * self.fps))
        if frame_id >= self.frame_count(): 
            # Frame index is OOB
            frame_id = None
        return frame_id

    # TODO: Slows down after a while when dealing with 100K frames
    def image_from_frame(self, frame_id):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        _, img = self.cap.read()
        return img

    """
    Compute frame count by parsing all
     - force=True will ensure that all readable frames are counted
    """
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
   
    """
    FFMPEG-Python for video metadata. 
     - More reliable than OpenCV for duration estimate
    """
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

    """
    Sample frame indices at n_FPS
     - n=1   for sampling at 1xFPS
     - n=10  for FPS/10 or 0.1xFPS sampling frame rate
     - n=0.1 for FPS/0.1 or 10xFPS sampling frame rate
    """
    def sample_at_n_fps(self, X, n=1):
        # Determine maxSecs as per original frame rate
        maxFrameNo = X.shape[0] - 1
        maxItems = int((maxFrameNo + 0.5) / (self.fps/n))
        frmItems = np.arange(maxItems + 1)
        indices_fps_round = np.rint(frmItems * (self.fps/n)).astype(int)
        return X[indices_fps_round]
        
    """
    Save frames sampled at FPS/n to a specified output directory
    """
    def save_frames_at_n_fps(self, n_fps, output_path):
        filepath, extension = os.path.splitext(self.videopath)
        basename = os.path.basename(filepath)
        if output_path is not None:
            filepath = os.path.join(output_path, basename)
        filepath = "{}-{}".format(filepath, extension[1:])
        os.makedirs(filepath, exist_ok=True)
        # Sample indices to be extracted as frames
        indices_nfps = set(self.sample_at_n_fps(np.arange(self.frame_count+1), n_fps))
        pprint("Sample count >>> {}, effective FPS:{:.4f} by using n_FPS:{} ".format(len(indices_nfps), self.fps/n_fps, n_fps ))
        # Initialization of frame index
        frame_id = 0      
        bad_frames = 0
        # Reset position of the video capture descriptor
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        with tqdm(total=self.frame_count) as pbar:
            while (self.cap.isOpened() and frame_id < self.frame_count):
                frame_id = int(round(self.cap.get(cv.CAP_PROP_POS_FRAMES)))
                ret, frame = self.cap.read()
                if not ret: # Bad frame
                    bad_frames += 1
                    if bad_frames >=10: 
                        pprint("{} multiple bad frames observed - {}/{}".format(bad_frames, frame_id, self.frame_count), log_type="WARN")
                        break
                elif frame_id in indices_nfps:
                    frame_path = os.path.join(filepath, "{}_frame_{:08d}.jpg".format(basename, frame_id))
                    cv.imwrite(frame_path, frame)
                    pbar.update(int(self.fps))
        pprint('Extracted {}/{} frames to directory {}'.format(len(indices_nfps), self.frame_count, filepath))
        return filepath, indices_nfps

        

#
# FFMPEG Utilities
#
class FrameExtractorFfmpeg:
    def __init__(self, videopath):
        """ FFMPEG utility to sample frames in a video  """
        self.videopath   = videopath
        self._json  = ffmpeg.probe(self.videopath)
        self.props  = self.get_props()
        self.fps    = self.props["fps"]
        self.width  = self.props["width"]
        self.height = self.props["height"]
        if "duration" in self.props.keys():
            self.frame_count = self.props["duration"] * self.props["fps"]
            pprint("FPS:{:.2f}, (Frames: {}, Duration {:.2f} s), \t Video:{} ".format(self.fps, self.frame_count, self.props["duration"], videopath))
        else:
            pprint("FPS:{:.2f}, (Video:{} ".format(self.fps, videopath))

    def get_props(self):
        """ Video ffprobe properties """
        video_props = { "lib": "ffmpeg" }
        if 'format' in self._json:
            if 'duration' in self._json['format']:
                video_props["duration"] = float(self._json['format']['duration'])   
        if 'streams' in self._json:
            # commonly stream 0 is the video
            for s in self._json['streams']:
                if 'duration' in s:
                    video_props["duration"] = float(s['duration'])
                if 'avg_frame_rate' in s and s['codec_type'] == 'video':
                    frame_rate = s['avg_frame_rate'].split('/')
                    video_props["fps"]    = float(frame_rate[0])/float(frame_rate[1])
                    video_props["width"]  = int(s['width'])
                    video_props["height"] = int(s['height'])
        if "duration" in video_props.keys():
            video_props["frame_count"] = int(video_props["duration"] * video_props["fps"])

        #pprint("FF Probe (raw): {}".format(self._json))
        return video_props

    def export_frames(self, output_dir, suffix=None):
        """ Read each video frame and write to an image in output directory """
        # for each frame in video output below image file
        if suffix is not None:
            framename = "{}-{}-%08d.jpg".format(get_clean_basename(self.videopath), suffix)
        else:    
            framename = "{}-%08d.jpg".format(get_clean_basename(self.videopath))
        # Command
        cmd="cd \"{}\" && ffmpeg -nostats -hide_banner -i \"{}\" {}".format(output_dir, self.videopath, framename)
        pprint("{}".format(cmd))
        process = subprocess.Popen(cmd, shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
        stdout, stderr = process.communicate()
        if stderr:
            pprint(stderr)
            return stderr
        return stdout

#
# OpenCV Utilities
#
class FrameExtractorOpencv:
    def __init__(self, videopath):
        """ OpenCV utility to sample frames in a video  """
        self.videopath   = videopath    
        self.cap         = cv.VideoCapture(videopath)
        self.fps         = self.cap.get(cv.CAP_PROP_FPS)
        self.width       = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.props       = self.get_props()
        pprint("FPS:{:.2f}, (Frames: {}, Duration {:.2f} s), \t Video:{} ".format(self.fps, self.frame_count, self.frame_count/self.fps, videopath))

    def image_from_frame(self, frame_id):
        """ Extract frame given the identifier """
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        _, img = self.cap.read()
        return img
    
    def image_frames_save(self, frame_ids, basename, output_dir):
        """ Extract list of frame given their identifiers """
        frame_files = []
        pbar = tqdm(total=self.frame_count)
        while self.cap.isOpened():
            pbar.update(1)
            frame_id = int(round(self.cap.get(cv.CAP_PROP_POS_FRAMES)))
            ret, img = self.cap.read()
            if ret and (frame_id in frame_ids):
                framename = "{}-{:08d}.jpg".format(basename, frame_id)
                frame_path = os.path.join(output_dir, framename)
                cv.imwrite(frame_path, img)  
                frame_files.append([os.path.join("images", framename)])
            elif not ret or frame_id >= self.frame_count:
                #print("Unreadable frame id:", frame_id)
                break
        pbar.close()
        return frame_files
    
    def count_frames_manual(self):
        """ Compute iteratively FPS given the full video file path """
        total, start = 0, time.time()               # loop over the frames of the video
        while True:
            (grabbed, frame) = self.cap.read()         # grab the current frame
            if not grabbed:                         # check to see if we have reached the end of the video
                break
            total += 1                              # increment the total number of frames read
        read_time = (time.time() - start)
        return total    

    def get_props(self):
        """ Video's duration in seconds, return a float number """
        video_props = { "lib": "opencv" }
        # Initialize a FrameExtractor to read video
        #video_props["duration"] = float(frame_extractor.frame_count / frame_extractor.fps)
        video_props["fps"]          = float(self.fps)
        video_props["width"]        = int(self.width)
        video_props["height"]       = int(self.height)
        video_props["frame_count"]  = int(self.frame_count)
        return video_props
    
    def export_frames(self, output_dir, suffix=None):
        """ Read each video frame and write to an image in output directory """
        frame_files = []
        pbar = tqdm(total=self.frame_count)
        while self.cap.isOpened():
            pbar.update(1)
            frame_id = int(round(self.cap.get(cv.CAP_PROP_POS_FRAMES)))
            ret, img = self.cap.read()
            if ret:
                # for each frame in video output below image file
                if suffix is not None:
                    framename = "{}-{}-{:08d}.jpg".format(get_clean_basename(self.videopath), suffix, frame_id)
                else:    
                    framename = "{}-{:08d}.jpg".format(get_clean_basename(self.videopath), frame_id)
                frame_path = os.path.join(output_dir, framename)
                cv.imwrite(frame_path, img)  
                frame_files.append(framename)
            elif not ret or frame_id >= self.frame_count:
                #pprint("End: Unreadable frame id:".format(frame_id))
                break
        pbar.close()
        self.frame_count  = len(frame_files)
        return frame_files

def get_frame_filename_by_video(vid_file, frame_id):
    return "{}-{:08d}.jpg".format(get_clean_basename(vid_file), frame_id)

def get_video_files(vid_folder, extensions=('.mp4', '.MP4', '.wmv', '.WMV', '.avi', '.AVI', '.mpg', '.MPG')):
    """ Look for video files in a folder """
    vid_files = []
    for root, dirnames, filenames in os.walk(vid_folder):
        for filename in filenames:
            if filename.endswith(extensions):
                vid_files.append(os.path.join(root, filename))
    return vid_files

#
# File IO Utility
#
def get_clean_basename(filename):
    """ Clean alphanumeric base filename string except '._-' """
    if filename is None:
        return None
    basename = os.path.basename(filename)
    clean_basename = "".join( x for x in basename if (x.isalnum() or x in "._-"))
    return clean_basename

def write_list_file(filename, rows, delimiter=','):
    """ Write list to file or append if it exists """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+") as my_csv:
        csvw = csv.writer(my_csv,delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC)
        csvw.writerows(rows)

def write_json_file(filename, data):
    """ Write dictionary data to JSON """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



#
# CVAT Utility
#
def extract_video_frames(video_loc, out_frames_path, suffix=None):
    """Extract all frames from video using FFMPEG (or OpenCV) to out_frames_path directory"""
    video_props = {}
    if os.path.isfile(video_loc):
        # Step 1: FFMPEG 
        frame_ext_ff = FrameExtractorFfmpeg(video_loc)
        video_props  = frame_ext_ff.get_props()
        # In case of header issues in videos. Causes an error in FFmpeg export fallback on CV2 extraction
        if video_props["fps"] > 0 and video_props["fps"] < 90 and "duration" in video_props.keys(): 
            out_log = frame_ext_ff.export_frames(out_frames_path, suffix)
        else:
            frame_ext_cv = FrameExtractorOpencv(video_loc)
            out_log = frame_ext_cv.export_frames(out_frames_path, suffix)
            video_props  = frame_ext_cv.get_props()
            frame_ext_cv.cap.release()
        # Read the files in frames directory for image count
        frame_files = [name for name in os.listdir(out_frames_path) if os.path.isfile(os.path.join(out_frames_path, name)) and name.startswith(get_clean_basename(video_loc))]
        video_props["image_count"] = len(frame_files)
        pprint("Video extraction properties: {}".format(video_props))
    return video_loc, video_props

def process_task_backup_archive(task_backup, output_dir):
    vid_info = []
    video_type_list = ('.mp4', '.MP4', '.wmv', '.WMV', '.avi', '.AVI', '.mpg', '.MPG')
    try:
        # Extract archive to a local folder
        ann_file = os.path.join(task_backup, 'annotations.json')
        if os.path.isfile(ann_file):
            # Lookup video file, annotations
            actual_vid_file = None
            for root, dirnames, filenames in os.walk(os.path.join(task_backup, 'data')):
                for filename in filenames:
                    if filename.endswith(video_type_list):
                        actual_vid_file = os.path.join(root, filename)
                        break
            vid_info = [actual_vid_file, None, None, None]
            if actual_vid_file is None:
                raise Exception("No video file found in archive")
            
            # Extract frames to path
            out_frames_path = os.path.join(task_backup, "frames")
            if output_dir is not None:
                out_frames_path = os.path.join(output_dir, "frames")
            os.makedirs(out_frames_path, exist_ok=True)
            video_loc, video_props = extract_video_frames(actual_vid_file, out_frames_path, suffix=os.path.basename(task_backup))
            vid_info = [video_loc, video_props, out_frames_path, actual_vid_file]
    except Exception as e:
        pprint("ERROR - {}, {}".format(task_backup, e))
        print(traceback.format_exc())
    return vid_info


"""
Extract Video frames at a specified frame rate
"""
def extract_video_frames(video_infile, n_fps, output_path):
    start = time.time()
    try:
        frmx = FrameExtractor(video_infile)  
        frmx.save_frames_at_n_fps(n_fps, output_path)
    except KeyboardInterrupt:
        pprint('Interrupted ctrl-c')
    finally:
        if frmx.cap:
            frmx.cap.release()
        pprint("Completed in {:.3f} Sec - {} ".format(time.time()-start, video_infile))
    return

def get_info(video_path:str):
    # read a informations
    cap = cv.VideoCapture(video_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height, length


def frame_resizing(width:int, height:int, frame_size) -> list:
    # resize the image frame
    if width > height:
        aspect_ratio = width / height
        if height >= frame_size:
            height = frame_size
        width = int(aspect_ratio*height)
    else:
        aspect_ratio = height / width
        if width >= frame_size:
            width = frame_size
        height = int(aspect_ratio*width)
    return [width, height]

"""
Visualize the frame with control/progress bar
"""
def plot_video_frames(video_infile):
    start = time.time()
    try:
        # Initialize the Video Frame Extractor for sampling frames
        frmx = FrameExtractor(video_infile)
        frame_count, fps = frmx.frame_count, frmx.fps
        pprint("FPS:{:.2f}, (Frames: {}, Duration {:.2f}), \t Video:{} ".format(fps, frame_count, frame_count/fps, video_infile))
        
        # OpenCV windowing functions
        cv_window_name = "FPS:{:.2f}, Frames:{}, Video:{}".format(fps, frame_count, os.path.basename(video_infile))
        def onCurrentFrameTrackbarChange(trackbarValue):
            pprint("Current Frames Value: {}".format(trackbarValue))
            pass
        cv.namedWindow(cv_window_name) 
        cv.createTrackbar('current-frame', cv_window_name, 1, frame_count, onCurrentFrameTrackbarChange)

        # Initialization of index and frame iteration
        frame_id = 0
        bad_frames = 0
        while frame_id < frame_count:
            # Get the frame given its index
            img_from_frame = frmx.image_from_frame(frame_id)
            if img_from_frame is None: # Bad frame, skip
                bad_frames += 1
                continue
            
            # Set the trackbar and show frame in opencv window
            cv.setTrackbarPos('current-frame', cv_window_name, frame_id)
            # Run detections and overlay image in window
            #detect.detect(img_from_frame, do_overlay=True)
            cv.imshow(cv_window_name, img_from_frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                pprint("Quit")
                break
            
            # Reset trackbar position to the current frame and proceed to next
            frame_id = cv.getTrackbarPos('current-frame', cv_window_name)
            frame_id = frame_id + 1
        else:
            print("All frames exhausted")
    except KeyboardInterrupt:
        pprint('Interrupted ctrl-c')
    finally:
        # The following frees up resources and closes all windows
        if frmx.cap:
            frmx.cap.release()
        cv.destroyAllWindows()
        pprint("Completed in {} Sec \t - {}, has {} bad frames ".format(time.time()-start, video_infile, bad_frames))

    return

def main(args):
    if args.video is not None:           # Process single video sample
        if args.save_frames is not None:
            if os.path.isfile(args.video):
                extract_video_frames(args.video, n_fps=float(args.save_frames), output_path=args.output_path)
            elif  os.path.isdir(args.video):
                glob_reg = "{}/**/*.{}".format(args.video, args.video_ext)
                for filename in glob.glob(glob_reg, recursive=True):
                    pprint("Processing: {}".format(filename))
                    extract_video_frames(filename, n_fps=float(args.save_frames), output_path=args.output_path)
        else:    
            # Visualize video 
            plot_video_frames(args.video)
    else:
        pprint("No --video file/folder argument provided: {}".format(args))
        
"""
Usage: 
    Download and extract frames from the video for assessment
    Check the correctness of Metadata

    python viz.py --video data/VIRAT_S_050201_05_000890_000944.mp4
    python viz.py --video data/VIRAT_S_010204_05_000856_000890.mp4 --save-frames 1
    python viz.py --output-path "./data/output" --save-frames 1 --video data/VIRAT_S_050201_05_000890_000944.mp4 
    python viz.py --output-path "./data/output" --save-frames 1 --video data/cctv_videopipe_2021030114.mp4 
    python viz.py --output-path "./data/output" --save-frames 0.1 --video-ext "mp4" --video data/
"""
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Download and process CCTV videos')
    parser.add_argument("--video",           type=str,   default=None)
    parser.add_argument("--video-ext",       type=str,   default="avi",  help="Video extensions: avi (default), mp4, etc")
    parser.add_argument("--save-frames",     type=str,   default=None,   help="default None for plotting frames. 1 for sampling at 1xFPS; 10 for 0.1xFPS; 0.1 for 10xFPS")
    parser.add_argument("--output-path",     type=str,   default=None)
    args=parser.parse_args()
    pprint(args)
    main(args)
    pprint("END")
