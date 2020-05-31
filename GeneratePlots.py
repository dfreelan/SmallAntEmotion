import colorsys

from JustPlayVideo import VideoPlayer
import os
import time

from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import cv2
import colorsys

class GeneratePlots():
    def __init__(self,video_path,start_time,end_time,keyframe_times,keyframe_points,
                 box_width=20,box_height=20,use_rgb=False,use_pca=True):
        self.player = VideoPlayer(video_path,start_time,end_time)
        self.keyframe_times = keyframe_times
        self.keyframe_points = keyframe_points
        self.box_width = box_width
        self.box_height = box_height
        self.use_rgb = use_rgb
        self.use_pca = use_pca

    def get_data_point(self,frame,frame_count):
        width = self.box_width
        height = self.box_height

        current_interval, current_interval_dest = self.get_current_interval_info(frame_count)

        center_x, center_y = self.perform_interpolation_between_frames(current_interval, current_interval_dest,
                                                                       frame_count)
        x_start = center_y - height // 2
        y_start = center_x - width // 2
        x_end = center_y + height // 2
        y_end = center_x + width // 2

        average_red = np.average(frame[x_start:x_end, y_start:y_end, :1])
        average_green = np.average(frame[x_start:x_end, y_start:y_end, 1:2])
        average_blue = np.average(frame[x_start:x_end, y_start:y_end, 2:3])

        frame[x_start:x_end, y_start:y_end] //= 2


        if(self.use_rgb):
            data_point = [average_red,average_green,average_blue]
        else:
            data_point = colorsys.rgb_to_hsv(average_red, average_blue, average_green)
        return data_point

    def generate_data_for_plot(self):
        time_arr = []
        data_arr = []
        frame,frame_count = self.player.get_next_frame()
        time_arr.append(frame_count)
        time_arr.append(data_arr)
    def perform_interpolation_between_frames(self, current_interval, current_interval_dest, frame_count):
        percent_through_interval = (frame_count - current_interval[0]) / (current_interval[1] - current_interval[0])
        center_y = (1 - percent_through_interval) * current_interval_dest[0][1] + percent_through_interval * \
                   current_interval_dest[1][1]
        center_x = (1 - percent_through_interval) * current_interval_dest[0][0] + percent_through_interval * \
                   current_interval_dest[1][0]
        center_y = int(center_y + .5)
        center_x = int(center_x + .5)
        return center_x, center_y

    def get_current_interval_info(self, frame_count):
        for i in range(len(self.keyframe_times)):
            if (self.keyframe_times[i] < frame_count):
                continue
            else:
                current_interval = [self.keyframe_times[i - 1], self.keyframe_times[i]]
                index = i
                break
        current_interval_dest = [self.keyframe_points[index - 1], self.keyframe_points[index]]
        return current_interval, current_interval_dest
