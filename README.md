# Advanced Lane Finding

#### Udacity Self-Driving Car Engineer Nanodegree Project

![Video Output][video-Standard-gif]

Project Summary
---

In this project, I wrote a software pipeline to identify the lane boundaries in a video.

The steps of this project are the following:

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
The code is written in python in an IPython notebook called `Project.ipynb`. The images for camera calibration are stored in the folder called `camera_cal`. The images in `test_images` are for testing the pipeline on single frames. Examples of the output from each stage of the pipeline is in the folder called `output_images`. The video called `project_video.mp4` is the video that the pipeline works well on. The `challenge_video.mp4` video is for testing the pipeline under somewhat trickier conditions. The `harder_challenge.mp4` video is another challenging video with sharp turns on the road.

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
[image-Polynomial]: ./output_images/straight_lines1_polynomial_output.jpg "Polynomial Fit"
[image-Pipeline]: ./output_images/test4_pipeline_output.jpg "Pipeline Output"
[video-Standard]: ./output_project_video.mp4 "Project Video"
[video-Standard-gif]: ./output_images/output_project_video.gif "Project Video GIF"
[image-video-screenshot]: ./output_images/video_screenshot.jpg "Project Video"


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

Here's an example of the output for this step:
![alt text][image2-out]

### 2. Thresholding with Color Transforms and Gradients

I used a combination of color and gradient thresholds to generate a binary image that clearly shows the lanes on the road.
The code for thresholding steps are at cells #9 through #18 in `Project.ipynb`. Here's an example of the output for this step.  

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

The code for my perspective transform includes a function called `unwarp()`, which appears in cell #5 in `Project.ipynb`.  This function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points with the following coordinates.

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 455      | 200, 0        |
| 695, 455      | 1080, 0      |
| 1125, 720     | 1080, 720      |
| 185, 720      | 200, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto test images and the warped counterparts to verify that the lines appear parallel in the warped images.

The transformation is applied using

``` python
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, img_size)
```

Here is an example of the output after perspective transformation.
![alt text][image-Perspective]

And here is the output after thresholding and perspective transformation.
![alt text][image-Top-Down]

### 4. Finding Lane-Line

I implemented an algorithms to identify lane lines pixels in a frame and fit 2nd order polynomials to each of the right and left lanes.

##### Lanes Finding Method: Peaks in the Histogram
I applied the following process to the thresholded warped image to map out the lane lines. I plotted a histogram of where the binary activations occur across the image. I first normalize each pixel value to 0-1 and calculate the histogram for the lower half of the image by calculating the sum across pixels vertically. The most prominent peaks in the this histogram are good indicators of the x-position of the base of the lane lines.

``` python
bottom_half = img[img.shape[0]//2:,:]
histogram = np.sum(bottom_half, axis=0)
```

##### Sliding Window
I used the x-position of the base of the lane lines at the bottom of the image as the starting point to where to search for the lines. From that point, I can use a sliding window, placed around the line centers to find and follow the lines up to the to of the frame.  

I split the histogram for the two lines. Then, I set up a few hyperparamters for the sliding windows.

``` python
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 150
# Set minimum number of pixels found to recenter window
minpix = 50
```

I then loop through each window and keep track of the activated pixels that fall into these windows.

##### Fit a Polynomial
After finding all the pixels that belong to each line, I fit a polynomial to the line.

``` python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

![alt text][image-Polynomial]

##### Skip the Sliding Windows Step Once The Lines Are Found
To increase efficiency in finding lines in a video, I don't start fresh on every frame. I search in a margin around the previous lane line position. If I lose track of the lines, I can go back to the sliding windows search to start over.


### 5. Measuring Curvature
Next, I calculated the radius of the 2nd order polynomial lane lines. I need to convert this value from pixel space to meter space. This requires to make assumptions about the length and width of the section of the lane in the real world. I assume that if I'm projecting a section of lane similar to the images I have used, the lane is about 30 meters long and 3.7 meters wide. Therefore, to convert from pixels to real-world meter measurements, I use:

``` python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension  
```

I then calculate the position of the vehicle with respect to center of the lane.

The code for curvature calculation and vehicle position from center is in cell #26 in the Jupyter Notebook.


### 6. Overlay Lanes on Image
Once I have the left and right line positions in warped space, I project them back down onto the road using inverse perspective matrix (`Minv`).


Here is an example image of my result plotted back down onto the road.

![alt text][image-Pipeline]

I implemented this step in cells #29 in my code in `Project.ipynb` in the function `process_image()`.  


### 7. Pipeline (Video)
After I tuned my pipeline on test images, I ran it on a video stream. To keep track of parameters and other information I defined `Line()` and `Boundaries()` classes in cell #28 to keep track of all the interesting parameters I measure from frame to frame and use that information to remove outliers and remove jitters in the results.

Here is an example of the final video.  The pipeline performed reasonably well on the entire project video. There were wobbly lines at some times that are ok but there were no catastrophic failures that would cause the car to drive off the road.


![Video Output][video-Standard-gif]

### 8. Improving results
The pipeline didn't perform well on the challenging videos. I applied these improvements to the pipeline to improve the results.
1. **Sanity Check**: When I calculate the new parameters for a new frame I check to make sure the parameters make sense. I considered these sanity checks for this purpose:

- Checking that the left and right lines have similar curvature
- Checking that the left and right lines are separated by approximately the right distance horizontally
- Checking that left and right lines are roughly parallel


2. **Reset**: If sanity checks reveal that the lane lines are problematic, I retain the previous positions from the frame prior and step to the next frame to search again. If I lose the lines for several frames in a row (currently set to 10), I start searching from scratch using a histogram and sliding window.

3. **Smoothing**: Line detections will jump around from frame to frame a bit and it can be preferable to smooth it to obtain a cleaner result. I use a weighted moving average low-pass filter to smooth the lane parameters over frames. I do weighted average of the new parameters and the parameters from the last frame.

---
### Discussion
This project shows that with just a camera and some simple image processing we can extract valuable information from an image or video that can potentially be fed into more sophisticated self-driving car algorithms.  


#### Shortcomings
The pipeline will likely degrade in performance in this scenarios:
1. Sharp turns and large curvatures
2. Going up or down the hill
3. Lane marketings that are not clearly visible, for example due to color contrast
4. Lane obstruction by other cars

#### Improvements
What can we do to make the pipeline more robust? Here are a few suggestions:
1. We can force constrains when fitting polynomials. In the current pipeline we are finding the left and right polynomials independent of each other and then later reject them if they don't pass the sanity checks. We can use these constraints when calculating the parameters. For example, can force the two polynomials to be parallel to each other when looking at the histogram and fitting polynomials.
2. We can calculate a confidence number for our estimation for each line and use that when calculating the new best fit by doing weighted average in the low pass filter. In this case, the weights will be a function of the confidence number.
3. We can improve warping by dynamically adjusting the source points (calibrating).
4. Low pass filtering introduces delay in response to sharp turns. We can adjust the los pass filter weights to make it faster when the car is moving on roads with sharp turns.  
