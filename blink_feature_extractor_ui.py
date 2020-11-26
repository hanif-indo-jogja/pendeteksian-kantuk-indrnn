#Reference:https://www.pyimagesearch.com/
from __future__ import print_function

import os

import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import*
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import ImageTk, Image
matplotlib.use("TkAgg")

from queue import Queue

import numpy as np
import cv2

from ear_extractor import EarExtractor
from blink_detector import BlinkDetector, BlinkDetectorRequest

def mytask(gui):

    def blink_detector(output_textfile, video_path):

        Q = Queue(maxsize=7)  

        ear_extractor = EarExtractor(video_path)
        blink_detector = BlinkDetector()  

        blink_count = 0

        number_of_frames = 0
        while gui.cancel != True:
            (is_no_frame_left, is_face_detected, frame, left_eye, right_eye, ear) = ear_extractor.extract()
            if is_no_frame_left:
                print('no frame left')
                print(number_of_frames)
                break

            Q.put(frame)

            if (is_face_detected == True):
                
                number_of_frames = number_of_frames + 1  # number of frames that face is detected

                bd_request = BlinkDetectorRequest(ear = ear, is_there_a_missing_ear = False)
                retrieved_blinks = blink_detector.track_ears(bd_request)

                if retrieved_blinks:
                    total_blinks = blink_detector.get_total_blinks()
                    blink_frame_freq = total_blinks / number_of_frames
                    
                    blink_count = blink_count + len(retrieved_blinks)
                    print()
                    print('=============')
                    print("Blink count: {0}".format(blink_count))
                    print()

                    for detected_blink in retrieved_blinks:
                        if (detected_blink.velocity > 0):
                            with open(output_file, 'ab') as f_handle:
                                f_handle.write(b'\n')
                                np.savetxt(f_handle,
                                    [total_blinks, blink_frame_freq * 100,
                                        detected_blink.amplitude, detected_blink.duration, detected_blink.velocity], 
                                    delimiter=', ', newline=' ',fmt='%.4f')

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                if Q.full():  #to make sure the frame of interest for the EAR vector is int the mid            
                    gui.update_ear(blink_detector.get_current_ear_series())
                    frame_minus_7 = Q.get()
                    gui.update_video(frame_minus_7)

                elif Q.full():
                    junk =  Q.get()

                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key != 0xFF:
                    break
            
            else:
                bd_request = BlinkDetectorRequest(ear = 0, is_there_a_missing_ear = True)
                blink_detector.track_ears(bd_request)
                
                while (Q.empty() != False):
                    frame_minus_7 = Q.get()
                    gui.update_video(frame_minus_7)

                Q.queue.clear()

                key = cv2.waitKey(1) & 0xFF

                if key != 0xFF:
                    break
            
            gui.update_gui()
        
        # do a bit of cleanup
        ear_extractor.release()



    #############
    ####Main#####
    #############

    output_path = 'output/blink_features'
    dataset_path = 'drowsiness_dataset'

    dataset_sublist = os.listdir(dataset_path)
    for sub_dataset_l1 in dataset_sublist:
        if gui.cancel == True:
            break

        path1 = dataset_path + '/' + sub_dataset_l1
        dirlist1 = os.listdir(path1)
        for sub_dataset_l2 in dirlist1:
            if gui.cancel == True:
                break

            path2 = path1 + '/' + sub_dataset_l2
            dirlist2 = os.listdir(path2)
            for sub_dataset_l3 in dirlist2:
                if gui.cancel == True:
                    break

                path3 = path2 + '/' + sub_dataset_l3
                filename = os.path.splitext(sub_dataset_l3)[0]

                output_leaf_folder = output_path + '/' + sub_dataset_l2 + '/'
                if not os.path.exists(output_leaf_folder):
                    os.makedirs(output_leaf_folder)

                output_file = ''
                if filename == '0':
                    output_file = output_leaf_folder + 'alert.txt'
                elif filename == '10':
                    output_file = output_leaf_folder + 'drowsy.txt'
                else:
                    continue

                print('')
                print('=============================================')
                print('========= Extracting Blink Features =========')
                print('current file    : ' + path3)
                print('output file     : ' + output_file)
                print('=============================================')
                print('')
                
                blink_detector(output_file, path3)
    
    cv2.destroyAllWindows()
    gui.quit()

        

class GUI():
    def __init__(self):
        self.gui = None
        self.ear_plot = None
        self.chart_canvas = None
        self.video_win = None
        self.image_container = None
        self.job = None
        self.cancel = False

    def run(self):
        self.create_gui()
        def task():
            mytask(self)
        
        self.job = self.gui.after(30, task)
        self.gui.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.gui.mainloop()

    def create_gui(self):
        main_window = tk.Tk()
        container = tk.Frame(main_window)
        container.pack(side="top", fill="both", expand = True)

        label = tk.Label(container, text="EAR", font=("Arial", 12, "bold"))
        label.pack(pady=10,padx=10)

        fig = plt.figure(figsize=(8,5), dpi=100)
        ear_plot = fig.add_subplot(111)
        EAR_series = np.zeros([13])
        frame_series = np.linspace(1,13,13)
        ear_plot.plot(frame_series, EAR_series)

        canvas = FigureCanvasTkAgg(fig, container)        
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        video_win = tk.Toplevel(main_window)
        video_win.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.image_container = tk.Label(video_win)
        self.image_container.pack(side="left", fill=tk.BOTH, padx=10, pady=10)

        self.gui = main_window
        self.ear_plot = ear_plot
        self.chart_canvas = canvas
        self.video_win = video_win
    
    def on_closing(self):
        self.cancel = True
        self.gui.quit()

    def update_gui(self):
        self.chart_canvas.draw()
        self.gui.update()

    def quit(self):
        self.gui.quit()
        self.gui.update()

    def update_ear(self, EAR_series):
        line = self.ear_plot.get_lines()[0]
        frame_series = line.get_xdata()
        self.ear_plot.cla()
        plt.ylim([0.0, 0.5])
        self.ear_plot.plot(frame_series,EAR_series)
    
    def update_video(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        width, height = frame.size
        maximum_height = 700
        new_ratio = maximum_height / height
        new_width = int(width * new_ratio)
        new_height = int(height * new_ratio)
        frame.thumbnail((new_width, new_height), Image.ANTIALIAS)

        frame = ImageTk.PhotoImage(frame)

        self.image_container.configure(image=frame)
        self.image_container.image = frame

GUI().run()

