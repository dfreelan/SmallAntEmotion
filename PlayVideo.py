import os
import time

from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import cv2
import colorsys
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, (1920,1080))
#cap = cv2.VideoCapture('E:\\Stepmania\\StepMania 5\\Songs\\AT-cm_730561477.mp4')
cap = cv2.VideoCapture('videos/vid.mp4')
import matplotlib.pyplot as plt
frame_count=-1
scatter_x= []
scatter_y=[]
time_start = 0
time_end = 0
hsv_arr = []
hsv_arr_background = []
trans_times = [[0,230],[230,260],[260,660],[660,740]]
trans_dests = [[[232,865],[232,865]],[[232,865],[232,857]],[[232,857],[232,857]],[[232,857+3],[171,845+3]]]



def get_hsv_from_coords():
    global hsv
    x_start = center_y - height // 2
    y_start = center_x - width // 2
    x_end = center_y + height // 2
    y_end = center_x + width // 2
    # x_start = 850
    # y_start = 202
    # x_end = 860
    # y_end = 246
    average_red = np.average(frame[x_start:x_end, y_start:y_end, :1])
    average_green = np.average(frame[x_start:x_end, y_start:y_end, 1:2])
    average_blue = np.average(frame[x_start:x_end, y_start:y_end, 2:3])
    frame[x_start:x_end, y_start:y_end] //= 2
    if(frame_count-60>=2):
        graph = np.asarray(Image.open("pngs/plot"+str(frame_count-60)+".png").resize((int(640/1.5+.5),int(480/1.5+.5)),Image.BILINEAR))[:,:,:3]
        if os.path.exists("pngs/plot_background"+str(frame_count-60)+".png"):
            graph2 = np.asarray(Image.open("pngs/plot_background"+str(frame_count-60)+".png").resize((int(640/1.5+.5),int(480/1.5+.5)),Image.BILINEAR))[:,:,:3]
            print(np.shape(graph))
            width2 = np.shape(graph)[0]
            height2 = np.shape(graph)[1]
            frame[500:width2+500,:height2] = graph
            frame[500-320:width2+500-320,:height2] = graph2

    # frame[x_start:x_end, y_start:y_end] //= 2
    # hsv = [average_red, 0, 0]
    hsv = colorsys.rgb_to_hsv(average_red, average_blue, average_green)

#start is 660
#end is 740
while(cap.isOpened()):
    ret, frame = cap.read()
    if( frame is None):
        break
    frame_count+=1
    if(frame_count<60*1):
        time_start = time.time()
        continue
    if(frame_count>845+60*4):
        time_end = time.time()
        break
    index = -1
    for i in range(len(trans_times)):
        if(trans_times[i][1]<frame_count):
            continue
        else:
            current_interval = trans_times[i]
            index = i
            break
    current_interval_dest = trans_dests[index]
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#202,849
    #246,870
    # frame[ 849:870,202:246, :]*=0
    #"171,845"
    # width = 30
    # height = 20
    # center_y = 857
    # center_x = 232
    # if(frame_count>660):
    #     #"171,845"
    #     count=np.min([frame_count,740])
    #     time_passed = count-660
    #     center_y+=time_passed*(845-center_y)/(740-660)
    #     center_x+=time_passed*(171-center_x)/(740-660)
    #     center_y=int(center_y+.5)
    #     center_x=int(center_x+.5)
    # get_hsv_from_coords()
    # hsv_arr.append(hsv)
    # mean = hsv[0]
    width = 24
    height = 20
    if(index!=-1):
        percent_through_interval = (frame_count-current_interval[0])/(current_interval[1]-current_interval[0])
        # print("currentinteval ")
        # print(current_interval)
        print("frame count")
        print(frame_count)
        # print("currentInterval[0]")
        # print(current_interval[0])
        # print("length of interval")
        # print((current_interval[1]-current_interval[0]))
        # print("percent")
        # print(percent_through_interval)
        # print("y")
        # print(current_interval_dest[0][1],percent_through_interval*current_interval_dest[1][1])
        # print("x")
        # print(current_interval_dest[0][0],percent_through_interval*current_interval_dest[1][0])
        center_y = (1-percent_through_interval)*current_interval_dest[0][1] + percent_through_interval*current_interval_dest[1][1]
        center_x = (1-percent_through_interval)*current_interval_dest[0][0] + percent_through_interval*current_interval_dest[1][0]
        center_y = int(center_y + .5)
        center_x=int(center_x+.5)
    else:
        print("frame count")
        print(frame_count)
        center_y = trans_dests[-1][1][1]
        center_x = trans_dests[-1][1][0]

        # center_x = (1-percent_through_interval)*current_interval[0] + percent_through_interval*current_interval[1]

    get_hsv_from_coords()
    hsv_arr.append(hsv)
    scatter_x.append(frame_count)
    scatter_y.append(hsv_arr)
    mean = hsv[0]

    width = 1000
    height = 800
    center_y = 500
    center_x = 1100
    get_hsv_from_coords()
    hsv_arr_background.append(hsv)
    

    # print(mean)
    # frame[202:246, 849:870, :]*=0
    # cv2.setMouseCallback('Drawing spline', mousePosition)
    # cv2.imshow('frame',frame[x_start:x_end, y_start:y_end])
    cv2.imshow('frame',frame)
    out.write(frame)
    # time.sleep(1/120)
    # print(type(frame))
    # print(frame)
    # print(type(frame))
    # print(np.shape(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
out.release()
fig, ax = plt.subplots()
ax.set_ylabel('Hue')
ax.set_xlabel('Time in frames')
ax.set_title('Smallant\'s hue over time')

print(time_start-time_end)
pca= PCA(1)
hsv_arr = np.asarray(hsv_arr)
print("shape of hsv_arr {}".format(np.shape(hsv_arr)))
pca.fit(hsv_arr)
transformed=pca.transform(hsv_arr)

transformed = np.add(transformed,-1*np.min(transformed)+1)
transformed = np.log(transformed)
print("shape of transformed " + str(np.shape(transformed)))
scatter = ax.scatter(x=scatter_x, y=np.squeeze(transformed[ : , :1]),
                     picker=True, s=1)

fig.savefig('plot.png')


fig2, ax2 = plt.subplots()
ax2.set_ylabel('Hue')
ax2.set_xlabel('Time in frames')
ax2.set_title('Control: BOTW hue over time')
print(time_start-time_end)
pca= PCA(1)
# hsv_arr_background = np.asarray(hsv_arr_background)
print("shape of hsv_arr {}".format(np.shape(hsv_arr_background)))
pca.fit(hsv_arr_background)
transformed=pca.transform(hsv_arr_background)
transformed = np.add(transformed,-1*np.min(transformed)+1)
transformed = np.log(transformed)
print("shape of transformed " + str(np.shape(transformed)))
scatter = ax2.scatter(x=scatter_x, y=np.squeeze(transformed[ : , :1]),
                     picker=True, s=1)
plt.show()

pca = PCA(1)
print("shape of hsv_arr {}".format(np.shape(hsv_arr)))
pca.fit(hsv_arr)
scatter_x_all  = scatter_x.copy()
scatter_y_all  = scatter_y.copy()
hsv_arr_all = hsv_arr.copy()
for i in range(2,len(scatter_x_all)):
    scatter_x=scatter_x_all[:i]
    scatter_y=scatter_y[:i]
    hsv=hsv_arr_all[:i]
    fig, ax = plt.subplots()
    ax.set_ylabel('Hue')
    ax.set_xlabel('Time in frames')
    ax.set_title('Smallant\'s hue over time')

    print(time_start-time_end)

    transformed = pca.transform(hsv)
    transformed = np.add(transformed,-1*np.min(transformed)+1)
    transformed = np.log(transformed)
    print("shape of transformed " + str(np.shape(transformed)))
    scatter = ax.scatter(x=scatter_x, y=np.squeeze(transformed[ : , :1]),
                         picker=True, s=1)

    fig.savefig('pngs/plot'+str(i)+'.png')
    plt.close("all")

pca = PCA(1)
print("shape of hsv_arr {}".format(np.shape(hsv_arr_background)))
pca.fit(hsv_arr_background)
scatter_x_all  = scatter_x.copy()
scatter_y_all  = hsv_arr_background.copy()
hsv_arr_all = hsv_arr_background.copy()
for i in range(2,len(scatter_x_all)):
    scatter_x=scatter_x_all[:i]
    scatter_y=scatter_y[:i]
    hsv=hsv_arr_all[:i]
    fig, ax = plt.subplots()
    ax.set_ylabel('Hue')
    ax.set_xlabel('Time in frames')
    ax.set_title('CONTROL: Background hue over time')

    print(time_start-time_end)

    transformed = pca.transform(hsv)
    transformed = np.add(transformed,-1*np.min(transformed)+1)
    transformed = np.log(transformed)
    print("shape of transformed " + str(np.shape(transformed)))
    scatter = ax.scatter(x=scatter_x, y=np.squeeze(transformed[ : , :1]),
                         picker=True, s=1)

    fig.savefig('pngs/plot_background'+str(i)+'.png')
    plt.close("all")