# Waveshare Stereo Vision


Camera calibration is important for doing accurate 3D measurement. 
This project explore techniques to calibrate [Wave share Stereo Camera](https://www.waveshare.com/wiki/IMX219-83_Stereo_Camera) using opencv Python for building a 3D sensor using Jetson Nano.

To get started with the basic of CSI camera, go to the [original repository](https://github.com/JetsonHacksNano/CSI-Camera).

## Calibration (Intrinsic && Extrinsic)

Purposes:

* correct artifacts of camera lenses such as distortion.
* estimate accurate focal length of the camera.
* output camera matrix to correct future images. 
* pose estimation
* get camera distance 
* get camera rotation and translation matrix


Usage:

capture calibration image series by runing the program:

```
python3 capture_image_series.py
```

press space to the capture images.

Run intrinsic calibration to get camera matrixs for both left and right cameras. This would do both extrinsic and intrinsic calibration.

```
python3 calibration.py
```

## Correspondence Search && Depth Map Calculation.

Using EXtrinsic and Intrinsic calibration result to calculate disparity
map and do depth reconstruction.

TODO

## 3D Reconstruction

TODO

## Structured Light 3D Measurement with Light Crafter

TODO