import cv2
import numpy as np

from filterpy.kalman import KalmanFilter

class Kalman():

    def __init__(self, height, width, dim_x=4, dim_z=2):
        # Inizialize Kalman Filter 
        self.prediction_image = np.zeros((height,width,3), np.uint8)
        self.kalman_filter = KalmanFilter(dim_x, dim_z)
        self.kalman_filter.F = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        self.kalman_filter.H = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0]])
        self.kalman_filter.P *= 1000
        self.kalman_filter.R = np.array([[0.1, 0],
                                    [0, 0.1]])
        self.kalman_filter.Q = np.array([[0.1, 0, 0, 0],
                                    [0, 0.1, 0, 0],
                                    [0, 0, 0.1, 0],
                                    [0, 0, 0, 0.1]])

    def makePrediction(self, x, y):
        # Predict the trajectory in the frame
        self.kalman_filter.x = np.array([x, y, 0, 0]).reshape(4, 1)
        z = np.array([[x + np.random.randn(1)[0] * 2],
                        [y + np.random.randn(1)[0] * 2]])

        # Kalman's Filter prediction
        self.kalman_filter.predict()

        # Update of the kalman's filter with the new position
        self.kalman_filter.update(z)
        filtered_state = self.kalman_filter.x
        filtered_x = filtered_state[0][0]
        filtered_y = filtered_state[1][0]
        self.prediction_image = cv2.circle(self.prediction_image, (int(x), int(y)), 2, (255, 0, 0), thickness=3)
        self.prediction_image = cv2.circle(self.prediction_image, (int(filtered_x), int(filtered_y)), 2, (0, 255, 0), thickness=3)

    def showPrediction(self):
        cv2.imshow("Prediction", self.prediction_image)
        cv2.waitKey(0)