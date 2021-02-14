import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import PIL.ImageDraw as ImageDraw
import RPi.GPIO as GPIO
from time import sleep

def create_category_index(label_path='coco_ssd_mobilenet/labelmap.txt'):
    f = open(label_path)
    category_index = {}
    for i, val in enumerate(f):
        if i != 0:
            val = val[:-1]
            if val != '???':
                category_index.update({(i-1): {'id': (i-1), 'name': val}})
            
    f.close()
    return category_index

def get_output_dict(image, interpreter, output_details, nms=True, iou_thresh=0.5, score_thresh=0.6):
    output_dict = {
                   'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                   'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                   'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                   'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]
                   }

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    #if nms:
    #    output_dict = apply_nms(output_dict, iou_thresh, score_thresh)
    return output_dict

'''def apply_nms(output_dict, iou_thresh=0.5, score_thresh=0.6):
    
    q = 90 # no of classes
    num = int(output_dict['num_detections'])
    boxes = np.zeros([1, num, q, 4])
    scores = np.zeros([1, num, q])
    # val = [0]*q
    for i in range(num):
        # indices = np.where(classes == output_dict['detection_classes'][i])[0][0]
        boxes[0, i, output_dict['detection_classes'][i], :] = output_dict['detection_boxes'][i]
        scores[0, i, output_dict['detection_classes'][i]] = output_dict['detection_scores'][i]
    nmsd = tf.image.combined_non_max_suppression(boxes=boxes,
                                                 scores=scores,
                                                 max_output_size_per_class=num,
                                                 max_total_size=num,
                                                 iou_threshold=iou_thresh,
                                                 score_threshold=score_thresh,
                                                 pad_per_class=False,
                                                 clip_boxes=False)
    valid = nmsd.valid_detections[0].numpy()
    output_dict = {
                   'detection_boxes' : nmsd.nmsed_boxes[0].numpy()[:valid],
                   'detection_classes' : nmsd.nmsed_classes[0].numpy().astype(np.int64)[:valid],
                   'detection_scores' : nmsd.nmsed_scores[0].numpy()[:valid],
                   }
    return output_dict'''

def display(val, img):
    #if 76 in val['detection_classes'][0]:
        #i = val['detection_classes'][0].index(76)
        
    heigth, width = 480, 640

    start_point = (width*val['detection_boxes'][0][1], heigth*val['detection_boxes'][0][0]) 
    end_point = (width*val['detection_boxes'][0][3], heigth*val['detection_boxes'][0][2])

    return start_point, end_point
        
def make_and_show_inference(img, interpreter, input_details, output_details, category_index, nms=True, score_thresh=0.6, iou_thresh=0.5):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (300, 300), cv2.INTER_AREA)
    img_rgb = img_rgb.reshape([1, 300, 300, 3])

    interpreter.set_tensor(input_details[0]['index'], img_rgb)
    interpreter.invoke()
    
    output_dict = get_output_dict(img_rgb, interpreter, output_details, nms, iou_thresh, score_thresh)
    #print(output_dict['detection_boxes'])
    #print(output_dict['detection_classes'])

    p1, p2 = display(output_dict, img_rgb)
    return p1, p2

def angleSet(angle,pin):
    servo=GPIO.PWM(pin,50)
    servo.start(0)
    duty=angle/18+2
    GPIO.output(pin,True)
    servo.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(pin,False)
    servo.ChangeDutyCycle(duty)
    servo.stop()


# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="coco_ssd_mobilenet/detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

GPIO.setmode(GPIO.BOARD)
servo_pitch=40
servo_yaw=38
GPIO.setup(servo_pitch, GPIO.OUT)
GPIO.setup(servo_yaw, GPIO.OUT)
currrent_pitch=90
current_yaw=90
angleSet(currrent_pitch,servo_pitch)
angleSet(current_yaw,servo_yaw)

category_index = create_category_index()
input_shape = input_details[0]['shape']
cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    if ret:
        p1, p2 = make_and_show_inference(img, interpreter, input_details, output_details, category_index)
        print(p1, p2)
        box_center = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)

        img = cv2.rectangle(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0) , 2)
        
        cv2.imshow("image", img)

        if p1[0]<320 and p2[0]>320:
            print("Horizontal target achieved")
        else:
            if p1[0]>320:
                currrent_pitch=currrent_pitch-2
                angleSet(currrent_pitch,servo_pitch)
            elif p2[0]<320:
                currrent_pitch=currrent_pitch+2
                angleSet(currrent_pitch,servo_pitch)

        if p2[0]<240 and p2[0]>240:
            print("Vertical target achieved")
        else:
            if p2[0]>240:
                currrent_yaw=currrent_yaw+2
                angleSet(currrent_yaw,servo_yaw)
            elif p2[0]<240:
                currrent_yaw=currrent_yaw-2
                angleSet(currrent_yaw,servo_yaw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()

# frame center = 320, 240
