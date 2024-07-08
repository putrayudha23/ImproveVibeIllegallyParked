import cv2
import numpy as np
from lib_vibe import vibe_gray
import math

#> masukan class vibe_gray (untuk vibe pada video grayscale) ke variabel "vibe"
vibe = vibe_gray()

#> read video dan masukan ke variable "cap"
cap = cv2.VideoCapture('./ISLab/ISLab-12.mp4') #./iLIDS/iLDS_Easy.mp4

#> inisialisasi urutan frame dalam variabel "frame_index"
frame_index = 0

#> kernel untuk closing
kernel = np.ones((5,5), np.uint8)

#> inisialisasi urutan frame untuk statis region
statis_start = 1
statis_fpsx10 = 0
endindx = 0
fps = cap.get(cv2.CAP_PROP_FPS)
fps = math.ceil(fps)
gap = fps+(math.ceil(fps/3))+10 #2*fps #fps+(math.ceil(fps/2)) #fps #math.ceil(fps/2) # agar 1 sec menghasilkan 2 frame objek statis
segmentation_sum = []

# container untuk BBOX dan Waktu
bboxX1 = 0
bboxY1 = 0
bboxX2 = 0
bboxY2 = 0
bbox_time = fps*10
fpsbuffer = 0
time_count = 0

# untuk save video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
hasilvideo = cv2.VideoWriter('./save/ISLab-13.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)

#> loop (while) proses ke semua frame di video
while True:
    #> read frame pada video di variabel "cap"
    ret, frame = cap.read()
    if not ret:
        break

    #> Penanganan bayangan START /////////////////////////////////////////
    dilated_img = cv2.dilate(frame, np.ones((7,7), np.uint8)) #dilasi pada  R G B channel untuk mengurangi efek bayangan
    bg_img = cv2.medianBlur(dilated_img, 21) #
    diff_img = cv2.absdiff(frame, bg_img) #cv2.absdiff is a function which helps in finding the absolute difference between the pixels of the two image arrays
  
    gray_frame = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    # Penanganan bayangan END ///////////////////////////////////////////////
    
    rows,cols = gray_frame.shape
    
    #/////////////////////////////////////////////////////////////////////////// VIBE START
    #> inisialisasi background model vibe
    if frame_index == 0: # frame pertama
        vibe.AllocInit(gray_frame) # inisialisasi nilai-nilai dari frame pertama (gray)

    #> segmentasi vibe (background/foreground) pada frame gray menggunakan background model
    segmentation_map = vibe.Segmentation(gray_frame)

    #> update foreground < 20 pixel
    # vibe.UpdateLess20pxl(gray_frame, segmentation_map)

    bboxX1 , bboxY1, bboxX2, bboxY2 = vibe.UpdateLess20pxlSVM(gray_frame, segmentation_map)
    #/////////////////////////////////////////////////////////////////////////// VIBE END

    #> post processing
    segmentation_map = cv2.medianBlur(segmentation_map, 3) #> reduce noise using median blur
    segmentation_map = cv2.morphologyEx(segmentation_map, cv2.MORPH_CLOSE, kernel) # closing untuk menutup hole pada foreground
    statis_buffer = segmentation_map*0

    #Bounding Box dan hitung waktu START //////////////////////////////////////////////////////////////
    if bboxX1 == 0 and bboxY2 == 0 and bboxX2 == 0 and bboxY2 == 0:
        bbox_time = fps*10
        fpsbuffer = 0
        time_count = 10
    else:
        bbox_time += 1
        fpsbuffer += 1
        if fpsbuffer == fps:
            time_count += 1
            fpsbuffer = 0

    print(time_count)

    if bbox_time <= fps*60 :
        cv2.rectangle(frame, (bboxX1,bboxY1), (bboxX2,bboxY2), (0,255,0), 3)
        cv2.putText(frame, str(time_count), (bboxX1,bboxY1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 3)
    else:
        cv2.rectangle(frame, (bboxX1,bboxY1), (bboxX2,bboxY2), (0,0,255), 3)
        cv2.putText(frame, "Alarm", (bboxX1,bboxY1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 3)
    #Bounding Box dan hitung waktu END //////////////////////////////////////////////////////////////

    #save video
    hasilvideo.write(frame)

    #> menampilkan hasil
    cv2.imshow('Hasil', frame) #show actual frame
    cv2.imshow('Segmentation Frame vibe', segmentation_map) #show segmentasi vibe

    #next frame to input vibe
    frame_index += 1

    #next to statis start
    statis_start += 1

    #agar tetap pada fps*10
    if frame_index >= gap+2:
        statis_fpsx10 += 1
    if statis_fpsx10 % (fps*10) == 0 and statis_fpsx10 != 0: #(fps*10)
        statis_fpsx10 = 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# hasilvideo.release()
cv2.waitKey()
cv2.destroyAllWindows()