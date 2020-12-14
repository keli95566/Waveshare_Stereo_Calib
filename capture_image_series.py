import cv2
import os,sys
from stereo_camera import *
current_file_path = os.path.dirname(os.path.abspath(__file__))


def capture_images():
    if not os.path.exists(os.path.join(current_file_path, "images")):
        os.mkdir(os.path.join(current_file_path, "images"))
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            sensor_mode=3,
            flip_method=0,
            display_height=720,
            display_width=1280,
        )
    )
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            sensor_mode=3,
            flip_method=0,
            display_height=720,
            display_width=1280,
        )
    )
    right_camera.start()

    cv2.namedWindow("CSI Cameras", cv2.WINDOW_AUTOSIZE)

    if (
        not left_camera.video_capture.isOpened()
        or not right_camera.video_capture.isOpened()
    ):
        # Cameras did not open, or no camera attached

        print("Unable to open any cameras")
        # TODO: Proper Cleanup
        SystemExit(0)

    img_counter = 1
    while cv2.getWindowProperty("CSI Cameras", 0) >= 0 :
        
        _ , left_image=left_camera.read()
        _ , right_image=right_camera.read()
        camera_images = np.hstack((left_image, right_image))
        cv2.imshow("CSI Cameras", camera_images)

        # This also acts as
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break
        elif keyCode%256 == 32:
            # SPACE pressed
            img_name_l = os.path.join("images","left_{}.png".format(img_counter))
            img_name_r = os.path.join("images","right_{}.png".format(img_counter))

            cv2.imwrite(img_name_l, left_image)
            cv2.imwrite(img_name_r, right_image)

            print("{} written!".format(left_image))
            img_counter += 1

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    capture_images()
