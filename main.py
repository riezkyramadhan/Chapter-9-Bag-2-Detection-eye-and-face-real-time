import cv2

cam=cv2.VideoCapture(0)
cam.set(3, 660)
cam.set(4, 500)
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
EyeDectector = cv2.CascadeClassifier("haarcascade_eye.xml")

while True:
    retV,frame = cam.read()
    color_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face  = faceDetector.detectMultiScale(color_gray, 1.3, 5)
    eye   = EyeDectector.detectMultiScale(color_gray,1.3,5)
    
    for (x,y,w,h) in face :
        frame= cv2.rectangle(frame, (x,y), (x+w,y+h),(0,225,255),2)
        rec_face= color_gray[y:y+h,x:x+h]
    
    for (x1,y1,w1,h1) in eye :
        frame= cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1),(0,225,0),2)
        rec_meye= color_gray[y1:y1+h1,x1:x1+h1]

    cv2.imshow("Output", frame)
    close = cv2.waitKey(1)& 0xFF
    if close == 27 or close == ord("n"):
        break

cam.release()
cv2.destroyAllWindows()        

