import cv2

# something just to actually play the video
# something that just shows the plot
# something that saves out plots
# something that overlays the fonts onto the video

class VideoPlayer():

    #Video path: any old path to an mp4
    #start_vid is time in seconds you want to start showing the video at
    #end_vid is time in seconds when you want the videeo to stop
    def __init__(self, video_path="videos/vid.mp4", start_vid=0,end_vid=-1):
        self.video_path = video_path
        self.start_vid = start_vid
        self.end_vid = end_vid
        self.frame_count = 0

    #get the video capture from opencv.
    #this capture allowed you to iterate through frames
    #with a call to cap.read()
    def get_capture(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self.cap

    import matplotlib.pyplot as plt
    def play(self):
        self.get_capture()
        self.frame_count = -1
        while (self.cap.isOpened()):


            frame, _ = self.get_next_frame()
            #frame = self.get_next_frame()
            #if frame is non there is no new frame
            if frame is None:
                break

            #show that frame!
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #we done now. delete all the windows. and we don't need the video file open anymore
        self.cap.release()
        cv2.destroyAllWindows()

    def get_next_frame(self):
        # frame is just a numpy array!
        #read until we get to start point
        frame = None
        while self.frame_count+1 < self.start_vid * 60:
            self.cap.read()
            self.frame_count+=1
        ret, frame = self.cap.read()
        self.frame_count += 1
        # opencv likes to give us a null frame at the end
        # if we show it, we get an error
        if frame is None:
            return None

        # check to see if we cut here or not

        if not self.end_vid == -1 and self.frame_count > self.end_vid * 60:
            return None;
        # self.frame_count +=1
        return frame,self.frame_count


if __name__ == '__main__':
    player = VideoPlayer(video_path="videos/vid.mp4",start_vid=1,end_vid=18.08)
    player.play()