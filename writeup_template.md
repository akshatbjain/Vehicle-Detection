**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

The code for this step is contained in the code snippet `src/hog.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

I tried various combinations of parameters (trial and error method) and got optimum results using `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. All 3 channels were used to extract the features as that way we can make use of all the information available to us. `YCrCb` color space was also decided after some trial and error with different color spaces. I noticed that `YCrCb` helps detect white cars better.

I trained a linear SVM using the `hinge` loss function. The features were extracted and then shuffled (in order to avoid overfitting). The dataset was split into training and testing data with a 20% split. The model is saved as a pickle file. This way we can simply load the pickle file for our next run, thus avoiding retraining everytime we want to run our code. The script automatically looks if a pickle file is available in the `src` directory. If not, it trains the model from scratch and saves the pickle file in the `src` directory.

The code for this is available in `src/SVM_classifier.py` file.

###Sliding Window Search

The sliding window code can be found in `src/sliding_window.py` file. I use only the lower half of the image for extracting HOG features to improve speed and accuracy. The cars are only present in the lower half of the images and not in the sky thus we can avoid running our sliding windows in that part. I used a `ystart = 400` and `ystop = 656`.

The extracted HOG features are then subsampled for each sliding window. The `cell_per_block` parameter determines the overlap of each window in terms of cell distance. I tried a lot of different cell sizes and found that using a combination of multiple cell sizes worked the best in my case. Thus I use `scale = 1.0, 1.2, 1.4, 1.6, 1.8 and 2.0`.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on six scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

