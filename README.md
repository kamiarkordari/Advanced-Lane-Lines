# Advanced Lane Finding

#### Udacity Self-Driving Car Engineer Nanodegree Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

Project Summary
---

In this project, I wrote a software pipeline to identify the lane boundaries in a video.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Files
---

The images for camera calibration are stored in the folder called `camera_cal`.  

The images in `test_images` are for testing the pipeline on single frames.  

Examples of the output from each stage of the pipeline is in the folder called `output_images`.

The video called `project_video.mp4` is the video that the pipeline works well on.  

The `challenge_video.mp4` video is for testing the pipeline under somewhat trickier conditions.  

The `harder_challenge.mp4` video is another challenging video!

[//]: # (Image References)

[image1]: ./output_images/undistort_output.jpg "Undistorted"
[image2-in]: ./test_images/straight_lines1.jpg "Road test"
[image2-out]: ./output_images/straight_lines1_undistort_output.jpg "Road Undistorted"
[image3]: ./output_images/straight_lines1_thresholding_output.jpg "Binary Example"

[image-SobelX]: ./output_images/straight_lines1_sobel_x.jpg "Sobel X Gradient"

[image-SobelY]: ./output_images/straight_lines1_sobel_y.jpg "Sobel Y Gradient"

[image-Sobel-Magnitude]: ./output_images/straight_lines1_sobel_magnitude.jpg "Sobel Gradient Magnitude"

[image-Sobel-Direction]: ./output_images/straight_lines1_sobel_direction.jpg "Sobel Gradient Direction"

[image-Sobel-S]: ./output_images/straight_lines1_hls.jpg "S"

[image-Perspective]: ./output_images/straight_lines1_perspective_transform_output.jpg "Perspective Transformed"

[image-Top-Down]: ./output_images/straight_lines1_top_down_output.jpg "Top Down"

[video1]: ./project_video.mp4 "Video"


Details
---
### Camera Calibration

In this step we compute the camera matrix and distortion coefficients
to correct camera distortion.

The code for this step is contained in cells #2 through #4 of the IPython notebook `Project.ipynb`.  

Every camera distorts the images it captures. Since each camera's distortion can be different, we need to perform calibration to correct the image and make it undistorted. Usually this process is done by using chessboard images taken by the camera.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

Pipeline (single images)
---
### 1. Distortion Correction

Now, we use the camera parameters that we calculated in the previous step to undistort road image.

The code for this step is contained in cell #11 of `Project.ipynb`.  

Here's an example of the output for this step:
![alt text][image2-out]


### 2. Thresholding with Color Transforms and Gradients

I used a combination of color and gradient thresholds to generate a binary image that clearly shows the lanes on the road.
The code for thresholding steps are at cells #8 through #18 in `Project.ipynb`. Here's an example of the output for this step.  

![alt text][image3]

##### Sobel Gradients
Sobelx and Sobely are horizontal and vertical gradients (changes in color or darkness). I applied threshold on Sobel gradient of the image using cv2.Sobel().

``` python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  
```

I used Sobelx for the final pipeline with thresholds of 20 and 100.

Here's an example of thresholding the x and y Sobel gradients.  
![alt text][image-SobelX]
![alt text][image-SobelY]

##### Magnitude of the Sobel Gradients
For this I used thresholding on the square root of the combined squares of `sobelx` and `sobely`.  

``` python
gradmag = np.sqrt(sobelx**2 + sobely**2)
```

Here's an example of thresholding the magnitude of Sobel gradients.  
![alt text][image-Sobel-Magnitude]

##### Direction of the Gradients
For this I used thresholding on gradient direction.  

``` python
absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
```


Here's an example of the thresholding on the direction of Sobel directions.  
![alt text][image-Sobel-Direction]

##### HLS and Color thresholds
I extracted S channel of image representation in the HLS color space and then applied a threshold on its absolute value.

``` python
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]
```

Here's an example of thresholding on S-channel.  
![alt text][image-Sobel-S]



### 3. Perspective Transform

The code for my perspective transform includes a function called `unwarp()`, which appears in cell #5 in `Project.ipynb`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points with the following coordinates.

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 455      | 200, 0        |
| 695, 455      | 1080, 0      |
| 1125, 720     | 1080, 720      |
| 185, 720      | 200, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

The transformation is applied using

``` python
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, img_size)
```

Here is an example of the output after perspective transformation.
![alt text][image-Perspective]

And here is the output after thresholding and perspective transformation.
![alt text][image-Top-Down]

### 4. Identify Lane-Line Pixels

I implemented an algorithms to identify lane lines pixels in a frame and fit 2nd order polynomials to each of the right and left lanes.

The algorithm works differently for the first frame and subsequent frames.

##### Lanes in the First Frame
We apply the
1. Take only the bottom half of the image

##### Lanes in the Subsequent Frames



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
