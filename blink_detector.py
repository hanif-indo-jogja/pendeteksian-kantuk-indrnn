import numpy as np
import pickle
from scipy.ndimage.interpolation import shift

MIN_AMPLITUDE = 0.04    
EPSILON = 0.01  # for discrete derivative (avoiding zero derivative)

class Blink():
    def __init__(self):

        self.start = 0 #frame
        self.startEAR = 1
        self.peak = 0  #frame
        self.peakEAR = 1
        self.end = 0   #frame
        self.endEAR = 0
        self.amplitude = (self.startEAR + self.endEAR - 2 * self.peakEAR) / 2
        self.duration = self.end - self.start + 1
        self.EAR_of_FOI = 0 #FrameOfInterest
        self.values = []
        self.velocity = 0  #Eye-opening velocity

class BlinkDetectorRequest():
    def __init__(self, ear = 0, is_there_a_missing_ear = False):
        self.ear = ear
        self.is_there_a_missing_ear = is_there_a_missing_ear

class BlinkDetector():
    def __init__(self):
        self.EAR_count_max = 13
        self.EAR_series = np.zeros([self.EAR_count_max])
        self.EAR_count = 0
        self.retrieved_blinks = []
        self.current_blink = Blink()
        self.last_blink = Blink()
        self.counter4blinks = 0
        self.total_blinks = 0
        self.skip = False        
        self.reference_frame = 20
        self.blink_ready = False

        self.svm_blink_detector = pickle.load(open(
            'third_party_models/Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav', 'rb'))


    def __linear_interpolate(self, start, end, N):
        m = (end - start) / (N + 1)
        x = np.linspace(1, N, N)
        y = m * (x - 0) + start
        return list(y)

    def __ultimate_blink_check(self, last_blink): 
        retrieved_blinks = []
        missed_blinks = False
        values = np.asarray(last_blink.values)
        threshold = 0.4 * np.min(values) + 0.6 * np.max(values)   # this is to split extrema in highs and lows
        N = len(values)
        derivative = values[1:N] - values[0:N-1]    #[-1 1] is used for derivative
        i = np.where(derivative == 0)
        if len(i[0]) != 0:
            for k in i[0]:
                if k == 0:
                    derivative[0] = -EPSILON
                else:
                    derivative[k] = EPSILON * derivative[k-1]
        M = N - 1    # len(derivative)
        zero_crossing = derivative[1:M] * derivative[0:M-1]
        x = np.where(zero_crossing < 0)
        xtrema_index = x[0] + 1
        xtrema_EAR = values[xtrema_index]
        updown = np.ones(len(xtrema_index))        # 1 means high, -1 means low for each extremum
        updown[xtrema_EAR < threshold] = -1           #this says if the extremum occurs in the upper/lower half of signal
        # concatenate the beginning and end of the signal as positive high extrema
        updown = np.concatenate(([1],updown,[1]))
        xtrema_EAR = np.concatenate(([values[0]],xtrema_EAR,[values[N - 1]]))
        xtrema_index = np.concatenate(([0], xtrema_index,[N - 1]))

        updown_xero_crossing = updown[1:len(updown)] * updown[0:len(updown) - 1]
        jump_index = np.where(updown_xero_crossing < 0)
        number_of_blinks = int(len(jump_index[0]) / 2)
        selected_EAR_First = xtrema_EAR[jump_index[0]]
        selected_EAR_Sec = xtrema_EAR[jump_index[0] + 1]
        selected_index_First = xtrema_index[jump_index[0]]
        selected_index_Sec = xtrema_index[jump_index[0] + 1]
        if number_of_blinks > 1:
            missed_blinks = True
        if number_of_blinks == 0:
            print(updown, last_blink.duration)
            print(values)
            print(derivative)
        for j in range(number_of_blinks):
            detected_blink = Blink()
            detected_blink.start = selected_index_First[2 * j]
            detected_blink.peak = selected_index_Sec[2 * j]
            detected_blink.end = selected_index_Sec[2 * j + 1]

            detected_blink.startEAR = selected_EAR_First[2 * j]
            detected_blink.peakEAR = selected_EAR_Sec[2 * j]
            detected_blink.endEAR = selected_EAR_Sec[2 * j + 1]

            detected_blink.duration = detected_blink.end - detected_blink.start + 1
            detected_blink.amplitude = 0.5 * (detected_blink.startEAR-detected_blink.peakEAR) + 0.5 * (detected_blink.endEAR-detected_blink.peakEAR)
            detected_blink.velocity = (detected_blink.endEAR - selected_EAR_First[2 * j + 1]) / (detected_blink.end - selected_index_First[2 * j + 1] + 1) #eye opening ave velocity
            retrieved_blinks.append(detected_blink)

        return missed_blinks,retrieved_blinks

    def __get_frame_margin_btw_two_blinks(self, blink):
        if blink.duration > 15:
            return 8
        else:
            return 1

    def __is_a_blink_not_a_noise(self, blink):
        return (blink.peakEAR < blink.startEAR and blink.peakEAR < blink.endEAR 
            and blink.start < blink.peak and blink.amplitude > MIN_AMPLITUDE)

    def __is_amplitude_balanced(self, blink):
        return ((blink.startEAR - blink.peakEAR) > (blink.endEAR - blink.peakEAR) * 0.25 
            and (blink.startEAR - blink.peakEAR) * 0.25 < (blink.endEAR - blink.peakEAR))

    def __track_blinks(self, EAR, EAR_series, is_eyes_closed, reference_frame, current_blink, last_blink, counter4blinks, total_blinks, skip):
        blink_ready = False

        if is_eyes_closed == True:

            current_blink.values.append(EAR)
            current_blink.EAR_of_FOI = EAR    

            if counter4blinks > 0:
                skip = False
                
            if counter4blinks == 0:
                current_blink.startEAR = EAR    # EAR_series[6] is the EAR for the frame of interest(the middle one)
                current_blink.start = reference_frame - 6   # reference-6 points to the frame of interest which will be the 'start' of the blink

            counter4blinks += 1

            if current_blink.peakEAR >= EAR:    # deciding the min point of the EAR signal
                current_blink.peakEAR = EAR
                current_blink.peak = reference_frame - 6

        # otherwise, the eyes are open in this frame
        else:

            if counter4blinks < 2 and skip == False :           # Wait to approve or reject the last blink
                frame_margin_btw_2blinks = self.__get_frame_margin_btw_two_blinks(last_blink)

                if ((reference_frame - 6) - last_blink.end) > frame_margin_btw_2blinks:
                    if  (self.__is_a_blink_not_a_noise(last_blink) == True):
                        if(self.__is_amplitude_balanced(last_blink) == True):
                            blink_ready = True

                            missed_blinks, retrieved_blinks = self.__ultimate_blink_check(last_blink)

                            tmp_retrieved_blinks = retrieved_blinks
                            retrieved_blinks = []
                            for blink in tmp_retrieved_blinks:
                                if (blink.velocity > 0):
                                    retrieved_blinks.append(blink)

                            #####
                            total_blinks = total_blinks + len(retrieved_blinks) 
                            counter4blinks = 0
                            print("MISSED BLINKS= {}".format(len(retrieved_blinks)))

                            return retrieved_blinks, current_blink, last_blink, counter4blinks, total_blinks, skip, blink_ready
                        else:
                            skip = True
                            print('rejected due to imbalance')
                    else:
                        skip = True
                        print('rejected due to noise,magnitude is {}'.format(last_blink.amplitude))
                        print(last_blink.start < last_blink.peak)

            # if the eyes were closed for a sufficient number of frames (2 or more)
            # then this is a valid CANDIDATE for a blink
            if counter4blinks > 1:
                current_blink.end = reference_frame - 7  # reference-7 points to the last frame that eyes were closed
                current_blink.endEAR = current_blink.EAR_of_FOI
                current_blink.amplitude = (current_blink.startEAR + current_blink.endEAR - (2 * current_blink.peakEAR)) / 2
                current_blink.duration = current_blink.end - current_blink.start + 1

                frame_margin_btw_2blinks = self.__get_frame_margin_btw_two_blinks(last_blink)

                if (current_blink.start - last_blink.end) <= frame_margin_btw_2blinks + 1:  #Merging two close blinks
                    print('Merging...')
                    frames_in_between = current_blink.start - last_blink.end - 1
                    print(current_blink.start, last_blink.end, frames_in_between)
                    values_btw = self.__linear_interpolate(last_blink.endEAR, current_blink.startEAR, frames_in_between)
                    
                    last_blink.values = last_blink.values + values_btw + current_blink.values
                    last_blink.end = current_blink.end       
                    last_blink.endEAR = current_blink.endEAR
                    if last_blink.peakEAR > current_blink.peakEAR: 
                        last_blink.peakEAR = current_blink.peakEAR
                        last_blink.peak = current_blink.peak
                    last_blink.amplitude = (last_blink.startEAR + last_blink.endEAR - (2 * last_blink.peakEAR)) / 2
                    last_blink.duration = last_blink.end - last_blink.start + 1
                else:                            #Should not Merge (a Separate blink)

                    last_blink.values = current_blink.values    

                    last_blink.end = current_blink.end            
                    last_blink.endEAR = current_blink.endEAR

                    last_blink.start = current_blink.start        
                    last_blink.startEAR = current_blink.startEAR

                    last_blink.peakEAR = current_blink.peakEAR    
                    last_blink.peak = current_blink.peak

                    last_blink.amplitude = current_blink.amplitude
                    last_blink.duration = current_blink.duration

            counter4blinks = 0

        retrieved_blinks = []
        return retrieved_blinks, current_blink, last_blink, counter4blinks, total_blinks, skip, blink_ready

    def track_ears(self, blink_detector_request):
        if (blink_detector_request.is_there_a_missing_ear == True):
            self.EAR_series = np.zeros([13])
            self.EAR_count = 0
            self.current_blink = Blink()
            self.last_blink = Blink()
            self.counter4blinks = 0
            self.skip = False
            self.reference_frame = 20
            self.blink_ready = False

            return []

        ear = blink_detector_request.ear
        self.EAR_series = shift(self.EAR_series, -1, cval=ear)
        self.EAR_count = self.EAR_count + 1

        if (self.EAR_count >= self.EAR_count_max ):
            is_eyes_closed = self.svm_blink_detector.predict(self.EAR_series.reshape(1,-1))
            if self.counter4blinks == 0:
                self.current_blink = Blink()
                
            (self.retrieved_blinks, self.current_blink, self.last_blink, self.counter4blinks, self.total_blinks,
            self.skip, self.blink_ready) = (
                self.__track_blinks(self.EAR_series[6], self.EAR_series, is_eyes_closed, 
                    self.reference_frame, self.current_blink, self.last_blink,
                    self.counter4blinks, self.total_blinks, self.skip)
            )

            self.reference_frame = self.reference_frame + 1

            if (self.blink_ready == True):
                self.reference_frame = 20   # initialize to a random number to avoid overflow in large numbers
                self.skip = True
                self.last_blink.end = -10 # re initialization
                return self.retrieved_blinks

        return []

    def get_current_ear_series(self):
        return self.EAR_series
        
    def get_total_blinks(self):
        return self.total_blinks