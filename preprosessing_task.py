import numpy as np

FREQ_INDEX = 0
AMP_INDEX = 1
DUR_INDEX = 2
VEL_INDEX = 3

def unroll_in_time(in_data, window_size, stride):
    n = len(in_data)
    if n <= window_size:
        out_data = np.zeros([1, window_size, 4])
        out_data[0, -n:, :] = in_data
        return out_data
    else:
        N = ((n - window_size) // stride) + 1
        out_data = np.zeros([N, window_size, 4])
        for i in range(N):
            if i * stride + window_size <= n:
                out_data[i, :, :] = in_data[i * stride:i * stride + window_size, :]
            else:  # this line should not ever be executed because of the formula used above N is the exact time the loop is executed
                break

        return out_data

class BlinksNormalizer():
    def __init__(self, blinks_count, freq, amp, dur, vel, 
        freq_mean, freq_stddev, amp_mean, amp_stddev, 
        dur_mean, dur_stddev, vel_mean, vel_stddev
    ):
        self.blinks_count = blinks_count
        self.freq = freq
        self.amp = amp
        self.dur = dur
        self.vel = vel
        self.freq_mean = freq_mean
        self.freq_stddev = freq_stddev
        self.amp_mean = amp_mean
        self.amp_stddev = amp_stddev
        self.dur_mean = dur_mean
        self.dur_stddev = dur_stddev
        self.vel_mean = vel_mean
        self.vel_stddev = vel_stddev
    
    def get_normalized_blinks(self):
        return self.normalize_blinks(self.blinks_count, 
            self.freq, self.freq_mean, self.freq_stddev, 
            self.amp, self.amp_mean, self.amp_stddev, 
            self.dur, self.dur_mean, self.dur_stddev, 
            self.vel, self.vel_mean, self.vel_stddev)

    def normalize_blinks(self, blinks_count, 
        freq, freq_mean, freq_stddev, 
        amp, amp_mean, amp_stddev, 
        dur, dur_mean, dur_stddev, 
        vel, vel_mean, vel_stddev
    ):
        normalized_blinks = np.zeros([blinks_count, 4])

        normalized_freq = (freq[0:blinks_count] - freq_mean) / freq_stddev
        normalized_blinks[:, FREQ_INDEX] = normalized_freq

        normalized_amp = (amp[0:blinks_count]  - amp_mean) / amp_stddev
        normalized_blinks[:, AMP_INDEX] = normalized_amp

        normalized_dur = (dur[0:blinks_count]  - dur_mean) / dur_stddev
        normalized_blinks[:, DUR_INDEX] = normalized_dur

        normalized_vel = (vel[0:blinks_count]  - vel_mean)  / vel_stddev
        normalized_blinks[:, VEL_INDEX] = normalized_vel

        return normalized_blinks

    
class BlinksNormalizerFactory():

    @staticmethod
    def fromText(alert_text_path, drowsy_text_path):

        alert_blinks_normalizer = BlinksNormalizerFactory.get_alert_blinks_normalizer(alert_text_path)
        drowsy_blinks_normalizer = BlinksNormalizerFactory.get_drowsy_blinks_normalizer(
            drowsy_text_path, alert_blinks_normalizer)
        
        return alert_blinks_normalizer, drowsy_blinks_normalizer
    
    @staticmethod
    def get_alert_blinks_normalizer(alert_text_path):
        alert_freq, alert_amp, alert_dur, alert_vel = (
            BlinksNormalizerFactory.load_blink_features_from_text(alert_text_path))

        alert_blinks_count = len(alert_freq)
        bunch_size = alert_blinks_count // 3   # one third alert blinks used for baselining
        remained_size = alert_blinks_count - bunch_size

        (alert_freq_mean, alert_freq_stddev, alert_amp_mean, alert_amp_stddev, 
            alert_dur_mean, alert_dur_stddev, alert_vel_mean, alert_vel_stddev) = (
                BlinksNormalizerFactory.get_means_and_stddevs(
                    bunch_size, alert_freq, alert_amp, alert_dur, alert_vel))

        return BlinksNormalizer(remained_size, 
            alert_freq, alert_amp, alert_dur, alert_vel, 
            alert_freq_mean, alert_freq_stddev, alert_amp_mean, alert_amp_stddev, 
            alert_dur_mean, alert_dur_stddev, alert_vel_mean, alert_vel_stddev)
    
    @staticmethod
    def get_drowsy_blinks_normalizer(drowsy_text_path, alert_blinks_normalizer):
        drowsy_freq, drowsy_amp, drowsy_dur, drowsy_vel = (
            BlinksNormalizerFactory.load_blink_features_from_text(drowsy_text_path))

        drowsy_blinks_count = len(drowsy_freq)

        abn = alert_blinks_normalizer

        alert_freq_mean     = abn.freq_mean
        alert_freq_stddev   = abn.freq_stddev
        alert_amp_mean      = abn.amp_mean
        alert_amp_stddev    = abn.amp_stddev
        alert_dur_mean      = abn.dur_mean
        alert_dur_stddev    = abn.dur_stddev
        alert_vel_mean      = abn.vel_mean
        alert_vel_stddev    = abn.vel_stddev

        return BlinksNormalizer(drowsy_blinks_count,
            drowsy_freq, drowsy_amp, drowsy_dur, drowsy_vel, 
            alert_freq_mean, alert_freq_stddev, alert_amp_mean, alert_amp_stddev, 
            alert_dur_mean, alert_dur_stddev, alert_vel_mean, alert_vel_stddev)

    @staticmethod
    def get_means_and_stddevs(bunch_size, freq, amp, dur, vel):
        freq_mean = np.mean(freq[-bunch_size:])
        freq_stddev = np.std(freq[-bunch_size:])
        if freq_stddev == 0:
            freq_stddev = np.std(freq)

        amp_mean = np.mean(amp[-bunch_size:])
        amp_stddev = np.std(amp[-bunch_size:])
        if amp_stddev == 0:
            amp_stddev = np.std(amp)

        dur_mean = np.mean(dur[-bunch_size:])
        dur_stddev = np.std(dur[-bunch_size:])
        if dur_stddev == 0:
            dur_stddev = np.std(dur)

        vel_mean = np.mean(vel[-bunch_size:])
        vel_stddev = np.std(vel[-bunch_size:])
        if vel_stddev == 0:
            vel_stddev = np.std(vel)
        
        return freq_mean, freq_stddev, amp_mean, amp_stddev, dur_mean, dur_stddev, vel_mean, vel_stddev
    
    @staticmethod
    def load_blink_features_from_text(text_path):
        freq = np.loadtxt(text_path, usecols=1)
        amp = np.loadtxt(text_path, usecols=2)
        dur = np.loadtxt(text_path, usecols=3)
        vel = np.loadtxt(text_path, usecols=4)

        return freq, amp, dur, vel