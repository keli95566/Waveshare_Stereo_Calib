import cv2
import numpy as np
import os
import glob
import pickle
#from sklearn.preprocessing import normalize

current_file_path = os.path.dirname(os.path.abspath(__file__))

def save_obj(obj, name ):

    with open(os.path.join(current_file_path,"calib_result", name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(os.path.join(current_file_path,"calib_result", name + '.pkl'), 'rb') as f:

        return pickle.load(f)

def intrinsic_calib(images,
                    out_name, 
                    CHECKERBOARD = (6, 8), 
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    for fname in images:
        name =  os.path.basename(os.path.normpath(fname))
        gray = cv2.imread(os.path.join(fname),cv2.IMREAD_GRAYSCALE)
        print(name, gray.shape)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,None)
                                                #cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        print(ret)
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)
            cv2.imwrite(os.path.join(current_file_path, "calib_result",name), img)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h,w= gray.shape[:2]
    Omtx, roi= cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    res              = dict()
    res["mtx"]       = mtx
    res["dist"]      = dist
    res["rvecs"]     = rvecs
    res["tvecs"]     = tvecs
    res["shape"]     = gray.shape
    res["omtx"]      = Omtx
    res["roi"]       = roi
    res["objpoints"] = objpoints
    res["imgpoints"] = imgpoints

    save_obj(res, out_name)    
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    return mtx, dist, rvecs, tvecs, Omtx, roi

def stereo_calib(criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    """
    """
    if not os.path.exists((os.path.join(current_file_path, "calib_result", "left_intrinsic_calib.pkl"))):
        left_images  = glob.glob(os.path.join(current_file_path, "images","left_*"))
        intrinsic_calib(left_images, "left_intrinsic_calib")

    if not os.path.exists((os.path.join(current_file_path, "calib_result", "right_intrinsic_calib.pkl"))):
        right_images = glob.glob(os.path.join(current_file_path, "images", "right_*")) 
        intrinsic_calib(right_images, "right_intrinsic_calib")

    right_calib  = load_obj( "right_intrinsic_calib")
    left_calib   = load_obj( "left_intrinsic_calib")

    img_shape    = left_calib["shape"][::-1]
    objptsL      = left_calib["objpoints"]
    objptsR      = right_calib["objpoints"]
    objpoints    = objptsL 
    imgpointsL   = left_calib["imgpoints"]
    imgpointsR   = right_calib["imgpoints"]
    mtxL, mtxR   = left_calib["mtx"]  , right_calib["mtx"]
    distL, distR = left_calib["dist"] , right_calib["dist"]

    #print(criteria_stereo)
    #print(img_shape)
    #print( left_calib["shape"])
    #print(objptsL)
    #print(objptsR)
    #print(objptsL== objptsR)

    flags   = 0
    flags  |= cv2.CALIB_FIX_INTRINSIC

    #flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    #flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    #flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    #flags |= cv2.CALIB_FIX_ASPECT_RATIO
    #flags |= cv2.CALIB_ZERO_TANGENT_DIST
    #flags |= cv2.CALIB_RATIONAL_MODEL
    #flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    #flags |= cv2.CALIB_FIX_K3
    #flags |= cv2.CALIB_FIX_K4
    #flags |= cv2.CALIB_FIX_K5
    retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                            imgpointsL,
                                                            imgpointsR,
                                                            mtxL,
                                                            distL,
                                                            mtxR,
                                                            distR,
                                                            img_shape,
                                                            criteria_stereo,
                                                            flags= flags)

    # StereoRectify function
    rectify_scale= 0 # if 0 image croped, if 1 image nor croped
    RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                    img_shape, R, T,
                                                    rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
    # initUndistortRectifyMap function
    Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                 img_shape, cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
    Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                 img_shape, cv2.CV_16SC2)
    #*******************************************
    #***** Parameters for the StereoVision *****
    #*******************************************

    # Create StereoSGBM and prepare all parameters
    window_size = 3
    min_disp = 2
    num_disp = 130-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2)

    # Used for the filtered image
    stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

    # WLS FILTER Parameters
    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

if __name__=="__main__":
    if not os.path.exists(os.path.join(current_file_path, "calib_result")):
        os.mkdir(os.path.join(current_file_path, "calib_result"))

    #left_images  = glob.glob(os.path.join(current_file_path, "images","left_*"))
    #right_images = glob.glob(os.path.join(current_file_path, "images", "right_*")) 
    #intrinsic_calib(left_images, "left_intrinsic_calib")
   # intrinsic_calib(right_images, "right_intrinsic_calib")
    stereo_calib()