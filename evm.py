import cv2
from math import pi, exp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt, find_peaks
from sklearn import preprocessing
import time

#WIP

def gaussian_kernel(num_rows=3, num_columns=3, sigma=1.4):
    x_half = num_columns // 2
    y_half = num_rows // 2
    gaussian_matrix = np.zeros((num_rows,num_columns))
    coefficient = 1 / (2 * np.pi * (sigma**2))
    for y in range(-y_half, y_half + 1):
        for x in range(-x_half, x_half + 1):
            exponent = -(x**2 + y**2) / (2 * sigma**2)
            gaussian_matrix[y + y_half, x + x_half] = coefficient * exp(exponent)
    gaussian_matrix /= gaussian_matrix.sum()
    return gaussian_matrix

def convolution(kernel_matrix, padded_input_matrix):
    #Obtaining frequently referenced values to reduce excess calculations
    kernel_radius = len(kernel_matrix) // 2
    input_length_y = len(padded_input_matrix)
    input_length_x = len(padded_input_matrix[0])

    #Initializing the output matrix
    output_matrix = np.zeros((input_length_y - 2 * kernel_radius, input_length_x - 2 * kernel_radius))

    #These are to get the centre coordinate
    for y_matrix in range(input_length_y - 2 * kernel_radius):
        for x_matrix in range(input_length_x - 2 * kernel_radius):

            #These are to iterate across the kernel and multiply each element to perform the convolution
            sum = 0
            for y_kernel in range(len(kernel_matrix)):
                for x_kernel in range(len(kernel_matrix[0])):
                    sum+=(kernel_matrix[y_kernel][x_kernel]*padded_input_matrix[y_matrix+y_kernel][x_matrix+x_kernel])
            output_matrix[y_matrix][x_matrix] = sum

    return output_matrix

def gaussian_pyramid(matrix, kernel, level=1):
    pyramid = matrix
    for counter in range(level):
        #Calculating padding for convolutions
        matrix = pyramid
        padding = (len(kernel[0]) - 1)//2
        pad_width = ((padding, padding), (padding, padding), (0, 0)) 
        matrix = np.pad(matrix, pad_width=pad_width, mode='constant')

        #Applying the convolution to each different channel
        r_channel = convolution(kernel,matrix[:,:,0])
        g_channel = convolution(kernel,matrix[:,:,1])
        b_channel = convolution(kernel,matrix[:,:,2])

        #Assembling channels
        matrix = np.stack((r_channel,g_channel,b_channel),axis=2)

        halved_matrix_length = matrix.shape[0]//2
        halved_matrix_height = matrix.shape[1]//2
        halved_matrix = np.zeros((halved_matrix_length,halved_matrix_height,3))
        for i in range(halved_matrix_length):
            for j in range(halved_matrix_height):
                halved_matrix[i,j,:] = matrix[i*2,j*2,:]
        pyramid = halved_matrix
        print(pyramid.shape)
        print("Pyramid:",counter)

    return pyramid

def image_preprocessing(image_path):
    img = Image.open(image_path)
    matrix = np.array(img)

    #Cutting out the transparency channel
    matrix = matrix[:,:,:3]
    print(matrix.shape)
    return matrix

def read_from_video(filepath='face.mp4'):
    cap = cv2.VideoCapture(filepath)
    array = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        array.append(frame)
    cap.release()
    cv2.destroyAllWindows()
    array = np.array(array)
    return array

def image_to_file(image, filename='new_image.png'):
    cv2.imwrite(filename, image)

def video_to_file(matrix, filename='output.mp4'):
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (matrix.shape[2], matrix.shape[1]))
    for i in range(matrix.shape[0]):
        out.write(matrix[i])
    out.release()

def gray_to_file(matrix, filename='gray.mp4'):
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (matrix.shape[2], matrix.shape[1]), isColor=False)
    for i in range(matrix.shape[0]):
        frame = cv2.cvtColor(matrix[i].astype(np.uint8), cv2.COLOR_GRAY2BGR)
        grayscale_frame = frame[:, :, 0:1]
        out.write(grayscale_frame)
    out.release()

def cv2_downscale_frame(frame, levels=5):
    downscaled_frame = frame
    for _ in range(levels):
        downscaled_frame = cv2.pyrDown(downscaled_frame)
    return downscaled_frame

# processed_array = [0]*face.shape[0]
# kernel = gaussian_kernel(5,5,1)
# for i in range(face.shape[0]):
#     pyramid = gaussian_pyramid(face[i], kernel, 5)
#     processed_array[i] = pyramid
#     print(i)

# processed_array = np.array(processed_array)

#DOWNSCALING A VIDEO
# downscaled_faces = []
# for i in range(face.shape[0]):
#     downscaled_frame = cv2_downscale_frame(face[i], levels=5)
#     downscaled_faces.append(downscaled_frame)
# downscaled_faces = np.array(downscaled_faces)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  #Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def normalize_3d(matrix):
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[2]):
            one_dimensional = matrix[:,i,j]
            maximum, minimum = one_dimensional.max(), one_dimensional.min()
            matrix[:,i,j] = (matrix[:,i,j] - minimum)/(maximum - minimum)
    return(matrix)

def get_peaks(array):
    peak_locations = find_peaks(array)[0]
    peak_values = [array[x] for x in peak_locations]
    return (peak_locations, peak_values)

face = read_from_video('images/face.mp4')

downscaled_faces = []
for i in range(face.shape[0]):
    downscaled_frame = cv2_downscale_frame(face[i], levels=3)
    downscaled_faces.append(downscaled_frame)
face = np.array(downscaled_faces)

#Dummy values for demonstration
fs = 30  #Sampling frequency in Hz (frames per second)
f_low = 0.5 #Low cutoff frequency
f_high = 2 #High cutoff frequency
selected_channel = 1 #BGR = Green

filtered_data = np.empty((face.shape[0],face.shape[1],face.shape[2]))
for i in range(face.shape[1]):
    for j in range(face.shape[2]):
        one_pixel = face[:,i,j,selected_channel]
        filtered_data[:, i, j] = bandpass_filter(one_pixel, f_low, f_high, fs, order=5)
    print(i,face.shape[1])

#gray_to_file(filtered_data, 'images/output.mp4')

average_filtered_signal = np.mean(filtered_data, axis=(1, 2))
peak_locations, peak_values = get_peaks(average_filtered_signal)

#Heart Rate Calculations
number_of_frames = peak_locations[-1] - peak_locations[0]
number_of_heartbeats = len(peak_locations)
bpm = round(number_of_heartbeats*60/(number_of_frames/fs))

plt.figure(figsize=(8, 6))
plt.plot(average_filtered_signal,'g',label='Green Channel')
plt.scatter(peak_locations, peak_values, color='red', marker='x', label='Peaks')
plt.title(f'Colour Channel Intensity Over Time (Average {bpm} BPM)')
plt.xlabel('Frame')
plt.ylabel('Colour Intensity')
plt.legend()
plt.grid(True)
plt.show()

print("Average BPM:",round(bpm))
print()