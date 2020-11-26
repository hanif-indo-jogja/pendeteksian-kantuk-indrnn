from scipy.spatial import distance as dist

from imutils import face_utils

import imutils
import dlib

import pickle

import numpy as np
import cv2

import subprocess
import shlex
import json

class EarExtractor():

    def __init__(self, video_path):
        self.video_path = video_path
        self.video_stream = cv2.VideoCapture(self.video_path)
        self.rotation_degree = self.get_rotation(self.video_path)

        self.face_detector = dlib.get_frontal_face_detector()
        self.facial_landmarks_predictor = dlib.shape_predictor(
            'third_party_models/shape_predictor_68_face_landmarks.dat')

    def get_video_stream(self):
        return self.video_stream

    def get_rotation(self, file_path):
        print(file_path)
        """
        Function to get the rotation of the input video file.
        Adapted from gist.github.com/oldo/dc7ee7f28851922cca09/revisions using the ffprobe comamand by Lord Neckbeard from
        stackoverflow.com/questions/5287603/how-to-extract-orientation-information-from-videos?noredirect=1&lq=1

        Returns a rotation None, 90, 180 or 270
        """

        """
        -loglevel error: only show the rotate field, no other info.
        -select_streams v:0: process the first video stream (ignore if multiple video streams are present)
        -show_entries stream_tags=rotate: returns the rotate information from the input video
        -of default=nw=1:nk=1: use default output format and don't show anything else, i.e. no-wrappers (nw) and no key (nk)
        """
        cmd = "ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1"
        
        args = shlex.split(cmd)
        args.append(file_path)
        
        ffprobe_output = subprocess.check_output(args).decode('utf-8')

        rotation = 0
        if len(ffprobe_output) > 0: 
            ffprobe_output = json.loads(ffprobe_output)
            rotation = ffprobe_output

        return rotation

        # this "adjust_gamma" function directly taken from : https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def get_left_eye_coordinates(self, facial_landmarks):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        return facial_landmarks[lStart:lEnd]
    
    def get_right_eye_coordinates(self, facial_landmarks):
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        return facial_landmarks[rStart:rEnd]

    def eye_aspect_ratio(self, eye):
                # compute the euclidean distances between the two sets of
                # vertical eye landmarks (x, y)-coordinates
                A = dist.euclidean(eye[1], eye[5])
                B = dist.euclidean(eye[2], eye[4])

                # compute the euclidean distance between the horizontal
                # eye landmark (x, y)-coordinates
                C = dist.euclidean(eye[0], eye[3])

                ear = (A + B) / (2.0 * C)

                return ear

    def extract(self):
        (grabbed, frame) = self.video_stream.read()
        if (grabbed == True):

            frame = imutils.resize(frame, width=550)
            frame = imutils.rotate_bound(frame, self.rotation_degree)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = self.adjust_gamma(gray, gamma=1.5)

            rects = self.face_detector(gray, 0)
            if (np.size(rects) != 0):
                
                facial_landmarks = self.facial_landmarks_predictor(gray, rects[0])
                facial_landmarks = face_utils.shape_to_np(facial_landmarks)

                left_eye = self.get_left_eye_coordinates(facial_landmarks)
                right_eye = self.get_right_eye_coordinates(facial_landmarks)
                leftEAR = self.eye_aspect_ratio(left_eye)
                rightEAR = self.eye_aspect_ratio(right_eye)

                averaged_ear = (leftEAR + rightEAR) / 2.0

                ear = averaged_ear
                is_face_detected = True
                is_no_frame_left = False
                return is_no_frame_left, is_face_detected, frame, left_eye, right_eye, ear

            else:
                left_eye = []
                right_eye = []
                ear = 0
                is_face_detected = False
                is_no_frame_left = False
                return is_no_frame_left, is_face_detected, frame, left_eye, right_eye, ear
        
        self.video_stream.release()
        left_eye = []
        right_eye = []
        ear = 0
        frame = None
        is_face_detected = False
        is_no_frame_left = True
        return is_no_frame_left, is_face_detected, frame, left_eye, right_eye, ear

    def release(self):
        if (self.video_stream is not None):
            self.video_stream.release()