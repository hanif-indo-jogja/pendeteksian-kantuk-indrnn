#Reference:https://www.pyimagesearch.com/
import numpy as np

import os

from ear_extractor import EarExtractor
from blink_detector import BlinkDetector, BlinkDetectorRequest

def detect_blinks(output_textfile, video_path):
    ear_extractor = EarExtractor(video_path)
    blink_detector = BlinkDetector()

    number_of_frames = 0
    while True:
        (is_no_frame_left, is_face_detected, frame, left_eye, right_eye, ear) = ear_extractor.extract()
        if is_no_frame_left:
            print('no frame left')
            print(number_of_frames)
            break


        if (is_face_detected == True):
            
            number_of_frames = number_of_frames + 1  # number of frames that face is detected

            bd_request = BlinkDetectorRequest(ear = ear, is_there_a_missing_ear = False)
            retrieved_blinks = blink_detector.track_ears(bd_request)

            if retrieved_blinks:
                total_blinks = blink_detector.get_total_blinks()
                blink_frame_freq = total_blinks / number_of_frames
                
                for detected_blink in retrieved_blinks:
                    if (detected_blink.velocity > 0):
                        with open(output_file, 'ab') as f_handle:
                            f_handle.write(b'\n')
                            np.savetxt(f_handle,
                                [total_blinks, blink_frame_freq * 100,
                                    detected_blink.amplitude, detected_blink.duration, detected_blink.velocity], 
                                delimiter=', ', newline=' ',fmt='%.4f')
        
        else:
            bd_request = BlinkDetectorRequest(ear = 0, is_there_a_missing_ear = True)
            blink_detector.track_ears(bd_request)

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
            
            detect_blinks(output_file, path3)

    

