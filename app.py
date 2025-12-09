from ultralytics import YOLO,solutions
import cv2
from function.helper import extract_detections,create_mask
from function.helper_email import send_email
import numpy as np
import time
import threading







def detect_from_video(video_path,confiden=0.5,device='cuda'):
    cap = cv2.VideoCapture(video_path)
    model = YOLO('model/fire.pt').to(device)
    cv2.namedWindow("YOLOV11 DETECT FROM VIDEO",cv2.WINDOW_NORMAL)
    ## ตั่งค่าเริ่มต้น
    region_points =      np.array([[13, 19], [623, 19], [622, 350], [14, 351]])
    counter= solutions.ObjectCounter(
        reg_pts=region_points,
        names = model.names,
        draw_tracks=False,
        line_thickness=2,
        view_in_counts=False,
        view_out_counts=False,
    )
    alert_interval = 5  # กำหนดช่วงเวลาการแจ้งเตือน (วินาที)
    last_alert_time = 0  # เวลาที่แจ้งเตือนครั้งล่าสุด
    while cap.isOpened():
        success,frame = cap.read()
        if not success:
            print(" video error ")
            break 
        ### time ###
        current_time = time.time()  # เวลาปัจจุบัน
        time_remaining = max(0, int(alert_interval - (current_time - last_alert_time)))
        ### time ###
        
        
        ##mask frame by function
        masked_frame = create_mask(frame,region_points)  
        ## start track
        track = model.track(masked_frame,persist=True,show=False,verbose=False,conf=confiden)
        img = counter.start_counting(frame,track)
        
        result_data = extract_detections(track,model)
        for values in result_data:
            clsname = values['classname']
            cx, cy = values['center']
            x1, y1, x2, y2 = values['box']

            if clsname == "fire" or clsname == "other" or clsname=="smoke":
                if current_time - last_alert_time > alert_interval:
                    cv2.circle(img,(cx,cy),2,(255,0,255),5)
                    cv2.line(img,(10,10),(cx,cy),(0,255,0),2)
                    image_path = "detected.jpg"
                    cv2.imwrite(image_path,img)
                    print(f"!!!! Detect {clsname} !!!!!!!")

                    ### send email
                    threading.Thread(target=send_email, args=("CCTV-ALERT",f"พบเจอ {clsname}", image_path)).start()
                    ### send email

                    ### reset time
                    last_alert_time = current_time

        print(f"time to next alert: {time_remaining}s", end="\r")
        cv2.imshow('YOLOV11 DETECT FROM VIDEO',img)
        if cv2.waitkey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyALLWindows()


video = "video/fire3.mp4"

detect_from_video(video,0.3)