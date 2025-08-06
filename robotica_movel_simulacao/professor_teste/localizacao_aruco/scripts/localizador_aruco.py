#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf2_ros
import tf.transformations as tf_trans
from geometry_msgs.msg import TransformStamped

# Parâmetros de calibração da câmera 
# Matriz K da câmera
CAMERA_MATRIX = np.array([
    [679.040475, 0.0, 334.004969],
    [0.0, 682.925941, 216.061362],
    [0.0, 0.0, 1.0]
])

# Coeficientes de distorção da câmera
DISTORTION_COEFFS = np.array([0.130185, -0.217471, 0.002607, 0.016523, 0.0])

# Tamanho físico da tag
ARUCO_MARKER_SIZE = 0.21

# Dicionário dos marcadores
ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


class ArucoLocator:
    def __init__(self):
        # rospy.set_param('/use_sim_time', True)
        rospy.init_node('aruco_locator_real')

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Assina o tópico da imagem da câmera
        self.image_subscriber = rospy.Subscriber(
            '/p3dx/camera/image_raw',
            #'/usb_cam/image_raw',
            Image,
            self.image_callback
        )

        rospy.loginfo("Nó localizador Aruco iniciado")

    def image_callback(self, msg):
        try:
            # Converte a mensagem de imagem do ROS para OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detecta os marcadores ArUco na imagem
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                ARUCO_DICTIONARY,
                parameters=cv2.aruco.DetectorParameters_create()
            )

            # Se algum marcador for detectado
            if ids is not None:
                # Estima a pose de cada marcador detectado
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    ARUCO_MARKER_SIZE,
                    CAMERA_MATRIX,
                    DISTORTION_COEFFS
                )

                for i, marker_id in enumerate(ids):
                    # rvec e tvec dão a pose do marcador em relação à câmera
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]

                    marker_frame = f"aruco_marker_{marker_id[0]}"

                    # Publicar a transformação no TF
                    self.publish_transform(
                        tvec, rvec, marker_frame, "usb_cam", msg.header.stamp
                    )

                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DISTORTION_COEFFS, rvec, tvec, 0.1)
                    self.print_tag_pose(tvec, rvec, marker_id[0])

            # Mostra a imagem com os marcadores detectados
            cv2.namedWindow("Imagem detectada pela câmera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Imagem detectada pela câmera",500,500)
            cv2.imshow("Imagem detectada pela câmera", frame)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Erro no callback da imagem: {e}")

    def publish_transform(self, tvec, rvec, child_frame, parent_frame, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # Posição (vetor de translação)
        t.transform.translation.x = tvec[0]
        t.transform.translation.y = tvec[1]
        t.transform.translation.z = tvec[2]

        # Orientação (vetor de rotação)
        rotation_matrix = cv2.Rodrigues(rvec)[0]
        quaternion = tf_trans.quaternion_from_matrix(
            np.vstack((np.hstack((rotation_matrix, [[0], [0], [0]])), [0, 0, 0, 1]))
        )
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(t)

    def print_tag_pose(self, tvec, rvec, tag_id):
        rotation_matrix = cv2.Rodrigues(rvec)[0]
        quaternion = tf_trans.quaternion_from_matrix(
            np.vstack((np.hstack((rotation_matrix, [[0], [0], [0]])), [0, 0, 0, 1]))
        )

        rospy.loginfo(
            f"--- Tag ID: {tag_id} --- \n"
            f"Posição x,y,z: [{tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}]\n"
            f"Orientação x,y,z,w: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]\n"
        )


if __name__ == '__main__':
    try:
        ArucoLocator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

