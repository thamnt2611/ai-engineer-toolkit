import cv2
import time

# open video stream
print("haha")
cap = cv2.VideoCapture('rtsp://onvif:8PYhFPRL"mxh@10.6.8.31:554/live/3db7aaef-46a0-42d4-ae96-bd02690b3cad')
print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
    print(frame.shape)
    w = frame.shape[1]
    h = frame.shape[0]
    print(w, h)
    break
# cap.release()
# set video resolution
# cap.set(3, 640)
# cap.set(4, 480)

# set video codec
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('./test/video/poc.avi', fourcc, 20.0, (w, h))

# # start timer
# start_time = time.time()
# frame_num = 0
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         # out.write(frame)
#         frame_num += 1
#         print(frame_num)
#         if frame_num%200 == 0:
#             cv2.imwrite(f'test/test{frame_num}.jpg', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # release resources
cap.release()
# out.release()
