from ultralytics import YOLO,solutions
import cv2
from function.helper import extract_detections,create_mask
from function.helper_email import send_email
import numpy as np
import time
import threading

def detect_from_video(video_path,confiden=0.5,device='cuda'):
    cap = cv2.VideoCapture(video_path)
    model = YOLO('model/yolo11n.pt').to(device)
    model.predict(classes=[0])

    cv2.namedWindow("YOLOV11 DETECT FROM VIDEO", cv2.WINDOW_NORMAL)

    #พื้นที่ตรวจจับ
    region_points = np.array([[40, 11], [1863, 7], [1870, 1074], [46, 1066]])

    counter = solutions.ObjectCounter(
        reg_pts=region_points,
        name=model.names,
        draw_tracks=False,
        line_thickness=2,
        view_in_counts=False,
        view_out_counts=False,
    )

    alert_interval = 5
    last_alert_time = 0
    fall_threshold_ratio = 1.2 #ถ้า width / hieght > ค่านี้  อาจล้ม

    while cap.isOpened():
        success,frame = cap.read()
        if not success:
            print(" video error ")
            break
        
        current_time = time.time()
        time_remaining = max(0, int(alert_interval - (current_time - last_alert_time)))

        #หน้ากากพื้นที่
        masked_frame = create_mask(frame, region_points)

        #ตรวจจับ
        track = model.track(masked_frame, presist=True, show=False, verbose= False, conf=confiden)
        img = counter.start_counting(frame,track)

        #ดึงข้อมูล
        result_data = extract_detections(track, model)

        for values in result_data:
            clsname = values['classname']
            cx, cy = values['center']
            x1, y1, x2, y2 = values['box']
            width = x2 - x1
            height = y2 - y1

            #ตรวจจับคนล้ม
            if clsname == "person":

                ratio = width /height

                print(width,height,ratio)

                if ratio > fall_threshold_ratio:

                    if current_time - last_alert_time > alert_interval:

                        cv2.putText(img, "FALL DETECTED!", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0 , 0, 255), 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        # Save & send
                        from datetime import datetime
                        image_path = f"fall_detected.jpg"
                        cv2.imwrite(image_path, img)    
                        #image_path = "fall_detected.jpg"
                        #cv2.imwrite(image_path, img)
                        print(f"!!!! FALL DETECTED at {cx}, {cy} !!!!")
                        threading.Thread(target=send_email, args=("CCTV FALL ALERT", "ตรวจพบการหกล้ม", image_path)).start()
                        last_alert_time = current_time
            
        print(f"Time to next alert: {time_remaining}s", end="\r")
        cv2.imshow('YOLOV11 DETECT FROM VIDEO', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# รันระบบ
video = "video/fall1.mp4"
detect_from_video(video, 0.3)