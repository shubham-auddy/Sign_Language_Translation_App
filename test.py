import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2

def showText(image, text, width, height, fontScale):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (width, height)
    
    # Green color in BGR
    color = (0, 256, 25)
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.putText() method
    image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    return image



def predict(results):

    for finger in results.multi_hand_landmarks:
    
        #base or Wrist
        wrist = finger.landmark[0]
        base_x = wrist.x
        base_y = wrist.y

        row = []
                
        for landmark in finger.landmark:
            # print(base_x - landmark.x, base_y - landmark.y, sep=",  ")

            row.append(base_x - landmark.x)
            row.append(base_y - landmark.y)



    # Inference test
    predict_result = model.predict(np.array([row]))
    # print(np.squeeze(predict_result))
    # print(np.argmax(np.squeeze(predict_result)))
    return labels[np.argmax(np.squeeze(predict_result))]



# Loading the saved model
model = tf.keras.models.load_model('model.hdf5')

#label
labels =['Withdraw', 'Diposit']

# Initializing for hand detection
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()

        # Flip on horizontal
        frame = cv2.flip(frame, 1)

        #convert Color profile
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detections
        results = hands.process(image)
        
        # If detected
        if results.multi_hand_landmarks:
            # Add indicator
            frame = showText(frame, str(predict(results)), 30, 50, 1)
            
            
            cv2.imshow('Hand Tracking', frame)
        else:
            cv2.imshow('Hand Tracking', frame)



        # If 'q' pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            

cap.release()
cv2.destroyAllWindows()