import cv2
import time
from pose_module import MultiPerDetector
from detection.person_det_media import real_det

num_person = 1
rp = real_det()
mpg = MultiPerDetector(rp,model_path='trainer/models/action10_9_np3.tflite',num_person=num_person)

# capture the video
cap = cv2.VideoCapture(0+ cv2.CAP_DSHOW)
# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
start = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        try:
            image, gestures = mpg.multi_per_gesture(frame,sort="area",viz=True)
            for gesture in gestures:
                print("gesture:",gesture[0], "Prob:", str(gesture[1]),"id:",gesture[2], "bbox:",gesture[3], "fingers:", gesture[4])
        except Exception as e:
            print(e)
            image = frame.copy()
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)
        start = end
        cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
        cv2.imshow('image', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
  
cap.release()
cv2.destroyAllWindows()