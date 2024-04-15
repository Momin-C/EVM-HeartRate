import cv2
import numpy as np
from collections import deque
import modify_view
import matplotlib.pyplot as plt
import constants as c
from scipy.signal import butter, filtfilt, find_peaks

selected_pixels = deque()
total = deque()

cap = cv2.VideoCapture(c.VIDEO_CAPTURE_INDEX)
FPS = cap.get(cv2.CAP_PROP_FPS)

#Dummy values for demonstration
fs = 30  #Sampling frequency in Hz (frames per second)
f_low = 0.5 #Low cutoff frequency
f_high = 2 #High cutoff frequency
selected_channel = 1 #BGR = Green

def bandpass_filter(data, lowcut=0.75, highcut=2.5, fs=30, order=5):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def get_peaks(array, fs=30, bpm_range=(40, 180)):
    freq_range = [bpm/60 for bpm in bpm_range]
    print(freq_range)
    
    distance_range = [fs / freq for freq in freq_range]
    print(distance_range)
    
    #0 or 1
    min_distance = distance_range[1]
    
    std_dev = np.std(array)
    dynamic_height = std_dev * 0.5  #Experiment more with height 
    
    peak_locations, _ = find_peaks(array, distance=min_distance, height=dynamic_height)
    
    peak_values = [array[x] for x in peak_locations]
    return (peak_locations, peak_values)


def calculateBPM(peak_locations):
    if len(peak_locations) < 2:
        return None 
    intervals = np.diff(peak_locations)/FPS 
    average_interval = np.mean(intervals)
    bpm = 60/average_interval 
    return bpm

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_pixels) == c.PIXELS_TO_SELECT:
            selected_pixels.popleft()
        selected_pixels.append([y,x])

if not cap.isOpened():
    print("ERROR: Can't open video")
    exit()

cv2.namedWindow(c.FRAME_NAME)
cv2.setMouseCallback(c.FRAME_NAME, click_event)

FRAMES_SELECTING = 30*10
section_count = 150
one_section = deque()
just_selected = False
bl, tr = None, None
size = 0

while True:
    result, frame = cap.read()
    if not result:
        print("ERROR: Can't open video")
        break
    
    frame = cv2.flip(frame, c.FLIP_HORIZONTAL)

    if len(selected_pixels) == 2 and not just_selected:
        bl, tr = modify_view.ModifyView.assign_rectangle_corners(selected_pixels)
        size = (tr[0]-bl[0])*(bl[1]-tr[1])
        just_selected = True
    if just_selected and len(one_section) >= section_count:
        y = bandpass_filter(one_section)
        peak_locations, peak_values = get_peaks(y)
        print(calculateBPM(peak_locations))
        one_section.clear()
    elif just_selected:
        crop = frame[bl[0]:tr[0], tr[1]:bl[1]][c.CHANNEL_OF_INTEREST]
        extracted_sum = np.mean(crop)
        one_section.append(extracted_sum)

    cv2.imshow(c.FRAME_NAME, frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()