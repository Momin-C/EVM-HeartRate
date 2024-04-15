import cv2
from constants import RECTANGLE_SIZE, GREEN, CHANNEL_OF_INTEREST

class ModifyView:
    def __init__(self):
        return
    
    @staticmethod
    def assign_rectangle_corners(values):
        bl = min(values)
        tr = max(values)
        return bl, tr
    
    @staticmethod
    def display_selected_pixels(frame, selected_pixels):
        count = len(selected_pixels)
        extracted_sum = 0
        index = 0
        for y,x in selected_pixels:
            extracted_sum+=frame[y][x][CHANNEL_OF_INTEREST]
            top_left = (x - RECTANGLE_SIZE, y - RECTANGLE_SIZE)
            bottom_right = (x + RECTANGLE_SIZE, y + RECTANGLE_SIZE)
            cv2.rectangle(frame, top_left, bottom_right, GREEN, thickness = cv2.FILLED)
            index+=1
        return frame, count, extracted_sum
    
    @staticmethod
    def display_text(frame, string):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        colour = GREEN
        thickness = 2
        image = cv2.putText(frame, string, (0,100), font, fontscale, colour, thickness, cv2.LINE_AA)
        return image