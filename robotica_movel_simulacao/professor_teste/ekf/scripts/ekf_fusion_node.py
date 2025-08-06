#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

class SimpleEKF:
    def __init__(self):
        rospy.init_node('ekf_fusion_node')

        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)
        rospy.Subscriber('/robot_pose_aruco', PoseWithCovarianceStamped, self.aruco_callback)

        self.pub = rospy.Publisher('/ekf_pose', PoseWithCovarianceStamped, queue_size=10)

        self.amcl_pose = None
        self.aruco_pose = None

    def amcl_callback(self, msg):
        self.amcl_pose = msg
        self.fuse_and_publish()

    def aruco_callback(self, msg):
        self.aruco_pose = msg
        self.fuse_and_publish()

    def fuse_and_publish(self):
        if self.amcl_pose is None or self.aruco_pose is None:
            return

        fused = PoseWithCovarianceStamped()
        fused.header.stamp = rospy.Time.now()
        fused.header.frame_id = "map"

        # Média das posições
        x1 = self.amcl_pose.pose.pose.position.x
        y1 = self.amcl_pose.pose.pose.position.y
        x2 = self.aruco_pose.pose.pose.position.x
        y2 = self.aruco_pose.pose.pose.position.y

        fused.pose.pose.position.x = (x1 + x2) / 2.0
        fused.pose.pose.position.y = (y1 + y2) / 2.0
        fused.pose.pose.position.z = 0.0

        fused.pose.pose.orientation = self.amcl_pose.pose.pose.orientation

        cov1 = np.array(self.amcl_pose.pose.covariance).reshape((6, 6))
        cov2 = np.array(self.aruco_pose.pose.covariance).reshape((6, 6))
        fused_cov = ((cov1 + cov2) / 2.0).flatten().tolist()

        fused.pose.covariance = fused_cov

        self.pub.publish(fused)

if __name__ == '__main__':
    try:
        SimpleEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

