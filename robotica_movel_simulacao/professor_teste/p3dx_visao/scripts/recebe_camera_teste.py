#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def callback(msg):
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    cv2.imshow("Câmera do Robô", frame)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('camera_viewer')
    rospy.Subscriber('/p3dx/camera/image_raw', Image, callback)
    rospy.spin()

