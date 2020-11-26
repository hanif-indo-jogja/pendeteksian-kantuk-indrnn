import numpy as np
import tkinter as tk
import tkinter.font as tkf
import tkinter.simpledialog as tkd
import tkinter.filedialog
from tkinter import *
import cv2
import imutils
from PIL import ImageTk, Image

from ndarray_helper import rotate_left

from ear_extractor import EarExtractor
from blink_detector import BlinkDetector, BlinkDetectorRequest
from drowsiness_detector import DrowsinessDetector

import json
import time

class MeanStd():
    def __init__(self):
        self.freq_mean      = 0
        self.freq_stddev    = 0
        self.amp_mean       = 0
        self.amp_stddev     = 0
        self.dur_mean       = 0
        self.dur_stddev     = 0
        self.vel_mean       = 0
        self.vel_stddev     = 0

class AlertStateVideoRecorderFrame(tk.Frame):
    def __init__(self, parent, gui):
        tk.Frame.__init__(self, parent)

        inner_frame = tk.Frame(self, width=500, height=300)
        inner_frame.pack(side="top", fill="both", expand=True)
        inner_frame.pack_propagate(0)

        label = tk.Label(inner_frame, text="Record a Video in Alert Condition (3 minutes long)", 
            font=("Arial", 12, "bold"), fg="#1a998e")
        label.pack(pady=10, padx=10)

        boldFont = tkf.Font (size=16, weight="bold")

        def open_file_dialog():
            filename =  tk.filedialog.askopenfilename(initialdir = "./",
                title = "Select file",filetypes = (("all files","*.*"),("all files","*.*")))
            gui.get_alert_state_features(filename)
            gui.show_choose_drowsy_video_windows()

        
        button = tk.Button(inner_frame, text="Choose a Video", command=open_file_dialog, 
            font=boldFont, width=50, fg="#1a998e")
        button.pack(side="top", padx=(30, 30), pady=(40, 0))

class AlertVideoFrame(tk.Frame):
    def __init__(self, parent, gui):
        tk.Frame.__init__(self, parent)

        inner_frame = tk.Frame(self)
        inner_frame.pack(side="left", fill=tk.BOTH, expand=True)
        inner_frame.pack_propagate(0)

        image_container = tk.Label(inner_frame)
        image_container.pack(side="left", fill=tk.BOTH, padx=10, pady=10)

        self.inner_frame = inner_frame
        self.image_container = image_container
        self.parent = parent
        self.gui = gui

        self.inner_frame_padx = 0
        self.inner_frame_pady = 10

    def update(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        width, height = frame.size
        maximum_height = 700
        new_ratio = maximum_height / height
        new_width = int(width * new_ratio)
        new_height = int(height * new_ratio)
        frame.thumbnail((new_width, new_height), Image.ANTIALIAS)        

        frame = ImageTk.PhotoImage(frame)
        
        self.gui.change_size(width=new_width + self.inner_frame_padx,
            height=new_height + self.inner_frame_pady)

        self.image_container.configure(image=frame)
        self.image_container.image = frame

    def show(self):
        self.tkraise()


class DrowsinessAlertDialog():
    def __init__(self, parent):
        window = tk.Toplevel(parent)
        container = tk.Frame(window, width=300, height=150)
        container.pack(fill="both", expand = True)
        container.pack_propagate(0)

        alert_font = tkf.Font(size=20, weight="bold")
        lb = Label(container, text="Ngantuk!",
            font=alert_font, width=80, fg="#f50f0f")
        lb.pack(side="top", padx=40, pady=(20, 12))

        button_font = tkf.Font(size=16, weight="bold")
        b = Button(container, text="OK", command=self.close,
            font=button_font, width=80, fg="#1a998e")
        b.pack(side="top", padx=100, pady=8)

        self.window = window

    def close(self):
        self.window.destroy()

class ChooseDrowsyVideoFrame(tk.Frame):
    def __init__(self, parent, gui):
        tk.Frame.__init__(self, parent)

        inner_frame = tk.Frame(self, width=500, height=300)
        inner_frame.pack(side="top", fill="both", expand=True)
        inner_frame.pack_propagate(0)

        label = tk.Label(inner_frame, text="Drowsiness Detector", font=("Arial", 12, "bold"), fg="#1a998e")
        label.pack(pady=10, padx=10)

        boldFont = tkf.Font (size=16, weight="bold")

        def open_file_dialog():
            filename =  tk.filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("all files","*.*"),("all files","*.*")))
            gui.detect_drowsiness(filename)
        
        button = tk.Button(inner_frame, text="Choose a Video", command=open_file_dialog, 
            font=boldFont, width=50, fg="#1a998e")
        button.pack(side="top", padx=(30, 30), pady=(40, 0))

    def show(self):
        self.tkraise()


class DrowsyVideoFrame(tk.Frame):
    def __init__(self, parent, gui):
        tk.Frame.__init__(self, parent)

        inner_frame = tk.Frame(self)
        inner_frame.pack(side="left", fill=tk.BOTH, expand=True)

        image_container = tk.Label(inner_frame)
        image_container.pack(side="left", fill=tk.BOTH, padx=10, pady=10)

        self.inner_frame = inner_frame
        self.image_container = image_container
        self.parent = parent
        self.gui = gui

        self.inner_frame_padx = 0
        self.inner_frame_pady = 10

    def update(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        width, height = frame.size
        maximum_height = 700
        new_ratio = maximum_height / height
        new_width = int(width * new_ratio)
        new_height = int(height * new_ratio)
        frame.thumbnail((new_width, new_height), Image.ANTIALIAS)        

        frame = ImageTk.PhotoImage(frame)
        
        self.gui.change_size(width=new_width + self.inner_frame_padx,
            height=new_height + self.inner_frame_pady)

        self.image_container.configure(image=frame)
        self.image_container.image = frame

    def show(self):
        self.tkraise()

class CoolDownTimer():
    def __init__(self, interval):
        self.interval = interval
        self.last_run = 0
        self.first_time_run = True

    def __call__(self,*args,**kwargs):
        now = time.time()
        if (self.first_time_run or (now - self.last_run) >= self.interval):
            self.last_run = time.time()
            self.first_time_run = False
            return True
        else:
            return False

class GUI():
    def __init__(self):
        self.main_window = None
        self.container = None
        self.alert_state_video_recorder_frame = None
        self.alert_video_frame = None
        self.choose_drowsy_video_frame = None
        self.drowsy_video_frame = None
        self.image_container = None
        self.dad = None

        self.record_alert_end_time = 180
        self.alert_mean_std = MeanStd()

        self.job = None
        self.cancel = False

        seq_len = 30
        hidden_size = 512
        dropout = 0.5
        num_layers = 2
        self.drowsiness_detector = DrowsinessDetector(seq_len, hidden_size, dropout, num_layers)

        self.is_cool_down_end = CoolDownTimer(6)

    def run(self):
        self.create_gui()

        self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.main_window.mainloop()

    def create_gui(self):
        main_window = tk.Tk()
        container = tk.Frame(main_window)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.pack_propagate(0)
        
        alert_state_video_recorder_frame = AlertStateVideoRecorderFrame(container, self)
        alert_state_video_recorder_frame.grid(row=0, column=0, sticky="nsew")

        alert_video_frame = AlertVideoFrame(container, self)
        alert_video_frame.grid(row=0, column=0, sticky="nsew")

        choose_drowsy_video_frame = ChooseDrowsyVideoFrame(container, self)
        choose_drowsy_video_frame.grid(row=0, column=0, sticky="nsew")

        drowsy_video_frame = DrowsyVideoFrame(container, self)
        drowsy_video_frame.grid(row=0, column=0, sticky="nsew")

        alert_state_video_recorder_frame.tkraise()

        self.main_window = main_window
        self.alert_state_video_recorder_frame = alert_state_video_recorder_frame
        self.alert_video_frame = alert_video_frame
        self.choose_drowsy_video_frame = choose_drowsy_video_frame
        self.drowsy_video_frame = drowsy_video_frame
        self.container = container
    
    def on_closing(self):
        self.cancel = True
        self.main_window.quit()

    def update_gui(self):
        self.main_window.update()

    def quit(self):
        self.main_window.quit()
        self.main_window.update()
    
    def get_alert_state_features(self, video_path):
        self.show_alert_video_window()

        ear_extractor = EarExtractor(video_path)
        blink_detector = BlinkDetector() 

        fps = ear_extractor.get_video_stream().get(cv2.CAP_PROP_FPS)

        freqs = []
        amps = []
        durs = []
        vels = []

        number_of_frames = 0
        count = 0
        while True:
            (is_no_frame_left, is_face_detected, frame, left_eye, right_eye, ear) = ear_extractor.extract()
            if is_no_frame_left:
                print('end frame')
                break

            self.alert_video_frame.update(frame)
            self.update_gui()

            if (is_face_detected == True):
                number_of_frames = number_of_frames + 1

                bd_request = BlinkDetectorRequest(ear = ear, is_there_a_missing_ear = False)
                retrieved_blinks = blink_detector.track_ears(bd_request)

                if retrieved_blinks:
                    total_blinks = blink_detector.get_total_blinks()
                    blink_frame_freq = total_blinks / number_of_frames * 100
                    
                    for detected_blink in retrieved_blinks:
                        freqs.append(round(blink_frame_freq, 4))
                        amps.append(round(detected_blink.amplitude, 4))
                        durs.append(round(detected_blink.duration, 4))
                        vels.append(round(detected_blink.velocity, 4))
            else:
                bd_request = BlinkDetectorRequest(ear = 0, is_there_a_missing_ear = True)
                retrieved_blinks = blink_detector.track_ears(bd_request)

            count += 1
            time_stamp = count / fps
            if ((time_stamp) >= self.record_alert_end_time):
                break
        
        freqs = np.array(freqs)
        amps = np.array(amps)
        durs = np.array(durs)
        vels = np.array(vels)

        alert_mean_std = self.alert_mean_std

        alert_mean_std.freq_mean = np.mean(freqs)
        alert_mean_std.freq_stddev = np.std(freqs)
        if alert_mean_std.freq_stddev == 0:
            alert_mean_std.freq_stddev = 0.000001

        alert_mean_std.amp_mean = np.mean(amps)
        alert_mean_std.amp_stddev = np.std(amps)
        if alert_mean_std.amp_stddev == 0:
            alert_mean_std.amp_stddev = 0.000001

        alert_mean_std.dur_mean = np.mean(durs)
        alert_mean_std.dur_stddev = np.std(durs)
        if alert_mean_std.dur_stddev == 0:
            alert_mean_std.dur_stddev = 0.000001

        alert_mean_std.vel_mean = np.mean(vels)
        alert_mean_std.vel_stddev = np.std(vels)
        if alert_mean_std.vel_stddev == 0:
            alert_mean_std.vel_stddev = 0.000001
        
    def detect_drowsiness(self, video_path):
        self.show_drowsy_video_window()

        ear_extractor = EarExtractor(video_path)
        blink_detector = BlinkDetector() 

        max_blink_per_sequence = 30
        feature_count = 4
        blink_sequence = np.zeros((max_blink_per_sequence, feature_count), dtype=np.float32)
        blink_count = 0
        stride = 2

        frameNth = 1
        number_of_frames = 0
        while True:
            (is_no_frame_left, is_face_detected, frame, left_eye, right_eye, ear) = ear_extractor.extract()
            if is_no_frame_left:
                break

            self.drowsy_video_frame.update(frame)
            self.update_gui()

            if (is_face_detected == True):
                number_of_frames = number_of_frames + 1

                bd_request = BlinkDetectorRequest(ear = ear, is_there_a_missing_ear = False)
                retrieved_blinks = blink_detector.track_ears(bd_request)

                if retrieved_blinks:
                    total_blinks = blink_detector.get_total_blinks()
                    blink_frame_freq = total_blinks / number_of_frames * 100
                    
                    for detected_blink in retrieved_blinks:
                        freq    =   round(blink_frame_freq, 4)
                        amp     =   round(detected_blink.amplitude, 4)
                        dur     =   round(detected_blink.duration, 4)
                        vel     =   round(detected_blink.velocity, 4)

                        blink = np.array([freq, amp, dur, vel])
                        blink_sequence = rotate_left(blink_sequence, 1)
                        blink_sequence[max_blink_per_sequence - 1] = blink
                        blink_count = blink_count + 1
                        
                        print('')
                        print('===========')
                        print('blink count: {0}'.format(blink_count))
                        print('')
                        if (self.is_cool_down_end() and (blink_count >= max_blink_per_sequence)):
                            blink_sequence_list = np.array([blink_sequence])
                            results = self.do_detect_drowsiness(blink_sequence_list)

            else:
                bd_request = BlinkDetectorRequest(ear = 0, is_there_a_missing_ear = True)
                retrieved_blinks = blink_detector.track_ears(bd_request)

            
            if (frameNth == 500):
                self.show_drowsiness_alert()
            elif (frameNth == 900):
                self.hide_drowsiness_alert()
                frameNth = 0

            frameNth = frameNth + 1

    def do_detect_drowsiness(self, blink_sequences):
        blink_sequences = blink_sequences.astype('float32')

        alert_mean_std = self.alert_mean_std
        train_mean_std = self.get_train_mean_std()
        
        self.z_score_normalization(blink_sequences, alert_mean_std)
        self.z_score_normalization(blink_sequences, train_mean_std)

        return self.drowsiness_detector.detect(blink_sequences)

    def perform_detect_drowsiness(self, blink_sequences):
        blink_sequences = blink_sequences.astype('float32')

        alert_mean_std = self.alert_mean_std
        train_mean_std = self.get_train_mean_std()
        
        self.z_score_normalization(blink_sequences, alert_mean_std)
        self.z_score_normalization(blink_sequences, train_mean_std)

        return self.drowsiness_detector.detect(blink_sequences)

    def z_score_normalization(self, blink, mean_std):
        blink[:,:,0] = ((blink[:,:,0].astype('float32') - np.float32(mean_std.freq_mean)) /
            np.float32(mean_std.freq_stddev))

        blink[:,:,1] = ((blink[:,:,1] - mean_std.amp_mean) /
            mean_std.amp_stddev)

        blink[:,:,2] = ((blink[:,:,2] - mean_std.dur_mean) /
            mean_std.dur_stddev)

        blink[:,:,3] = ((blink[:,:,3] - mean_std.vel_mean) /
            mean_std.vel_stddev)

    def get_train_mean_std(self):
        json_mean_std = []
        with open('train_mean_std.json', 'r') as f_handle:
            json_mean_std = json.loads(f_handle.read())

        train_mean_std = MeanStd()

        train_mean_std.freq_mean    = np.float32(json_mean_std['freq_mean'])
        train_mean_std.freq_stddev  = np.float32(json_mean_std['freq_stddev'])

        train_mean_std.amp_mean     = np.float32(json_mean_std['amp_mean'])
        train_mean_std.amp_stddev   = np.float32(json_mean_std['amp_stddev'])

        train_mean_std.dur_mean     = np.float32(json_mean_std['dur_mean'])
        train_mean_std.dur_stddev   = np.float32(json_mean_std['dur_stddev'])

        train_mean_std.vel_mean     = np.float32(json_mean_std['vel_mean'])
        train_mean_std.vel_stddev   = np.float32(json_mean_std['vel_stddev'])

        return train_mean_std

    def partition_by_time_window(self, blink, window_size, stride):
        new_blink = []
        blink_length = len(blink)

        if (blink_length <= window_size):
            new_blink = np.zeros([1, window_size, 4])
            new_blink[0, -blink_length:, :] = blink
        else:
            n = ((blink_length - window_size) // stride) + 1
            new_blink = np.zeros([n, window_size, 4])
            for i in range(n):
                if i * stride + window_size <= blink_length:
                    new_blink[i, :, :] = blink[i * stride:window_size + (i * stride), :]
                else:
                    break

        return new_blink_features

    def show_choose_drowsy_video_windows(self):
        self.choose_drowsy_video_frame.show()

    def show_alert_video_window(self):
        self.alert_video_frame.show()

    def show_drowsy_video_window(self):
        self.drowsy_video_frame.show()
    
    def show_drowsiness_alert(self):
        self.dad = DrowsinessAlertDialog(self.main_window)
    
    def hide_drowsiness_alert(self):
        self.dad.close()

    def update_drowsy_video(self, frame):
        self.drowsy_video_frame.update(frame)

    def change_size(self, width, height):
        self.main_window.geometry("{0}x{1}".format(width, height))

GUI().run()       

