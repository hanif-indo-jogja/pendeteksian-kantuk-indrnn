import os
import numpy as np
from preprosessing_task import BlinksNormalizerFactory, unroll_in_time

def Preprocess(path, window_size, stride):
    #path is the address to the folder of all subjects, each subject has three txt files for alert and sleepy levels
    #window_size decides the length of blink sequence
    #stride is the step by which the moving windo slides over consecutive blinks to generate the sequences
    #output=[N,T,F]
    
    output = []
    labels = []
    out_test = []
    labels_test = []

    subject_list = os.listdir(path)
    subject_list.sort()   

    all_subject_normalized_blinks = []
    all_subject_labels = []
    for ID,folder in enumerate(subject_list):
        print("#########\n")
        print(str(ID)+'-'+ str(folder)+'\n')
        print("#########\n")
        files_per_person = os.listdir(path + '/' + folder)

        alertTXT = path + '/' + folder + '/' + 'alert.txt'
        drowsyTXT = path + '/' + folder + '/' + 'drowsy.txt'

        alert_blinks_normalizer, drowsy_blinks_normalizer =  BlinksNormalizerFactory.fromText(alertTXT, drowsyTXT)

        alert_normalized_blinks = alert_blinks_normalizer.get_normalized_blinks()
        alert_blinks_unrolled = unroll_in_time(alert_normalized_blinks, window_size, stride)
        alert_labels = 0 * np.ones([len(alert_blinks_unrolled)], dtype=np.int32)            

        drowsy_normalized_blinks = drowsy_blinks_normalizer.get_normalized_blinks()
        drowsy_blinks_unrolled = unroll_in_time(drowsy_normalized_blinks, window_size, stride)
        drowsy_labels = 1 * np.ones([len(drowsy_blinks_unrolled)], dtype=np.int32)

        tempX = np.concatenate((alert_blinks_unrolled, drowsy_blinks_unrolled),axis=0)
        tempY = np.concatenate((alert_labels, drowsy_labels), axis=0)

        all_subject_normalized_blinks.append(tempX)
        all_subject_labels.append(tempY)

    all_subject_normalized_blinks = np.array(all_subject_normalized_blinks)
    all_subject_labels = np.array(all_subject_labels)

    train_ratio = 8
    eval_ratio = 1
    test_ratio = 1
    total_ratio = train_ratio + eval_ratio + test_ratio
    subject_count = len(all_subject_normalized_blinks)
    train_subject_count = int(subject_count * (train_ratio / total_ratio))
    eval_subject_count = int(subject_count * (eval_ratio / total_ratio))
    test_subject_count = int(subject_count * (test_ratio / total_ratio))

    retrieved_subject_count = 0
    out = all_subject_normalized_blinks[retrieved_subject_count:retrieved_subject_count + train_subject_count]
    labels = all_subject_labels[retrieved_subject_count:retrieved_subject_count + train_subject_count]
    out = np.array(np.concatenate(out))
    labels = np.array(np.concatenate(labels))

    retrieved_subject_count = retrieved_subject_count + train_subject_count
    out_val = all_subject_normalized_blinks[retrieved_subject_count:retrieved_subject_count + eval_subject_count]
    labels_val = all_subject_labels[retrieved_subject_count:retrieved_subject_count + eval_subject_count]
    out_val = np.array(np.concatenate(out_val))
    labels_val = np.array(np.concatenate(labels_val))

    retrieved_subject_count = retrieved_subject_count + eval_subject_count
    out_test = all_subject_normalized_blinks[retrieved_subject_count:retrieved_subject_count + test_subject_count]
    labels_test = all_subject_labels[retrieved_subject_count:retrieved_subject_count + test_subject_count]
    out_test = np.array(np.concatenate(out_test))
    labels_test = np.array(np.concatenate(labels_test))
    
    return out, labels, out_val, labels_val, out_test, labels_test

path ='output/blink_features'
window_size = 30
stride = 2
config_name = 'w30_s2'

blinks, labels, blinks_val, labels_val, blinks_test, labels_test = Preprocess(
    path, window_size, stride)

print('====== shape ======')
print(blinks.shape)
print(labels.shape)
print(blinks_val.shape)
print(labels_val.shape)
print(blinks_test.shape)
print(labels_test.shape)

base_path = 'output/preprocessed_blink_features/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

np.save(base_path + 'Blinks_' + config_name, blinks)
np.save(base_path + 'Labels_' + config_name, labels)

np.save(base_path + 'BlinksVal_' + config_name, blinks_val)
np.save(base_path + 'LabelsVal_' + config_name, labels_val)

np.save(base_path + 'BlinksTest_' + config_name, blinks_test)
np.save(base_path + 'LabelsTest_' + config_name, labels_test)

