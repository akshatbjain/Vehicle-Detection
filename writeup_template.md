**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_car_example.jpg
[image3]: ./output_images/HOG_notcar_example.jpg
[image4]: ./output_images/sliding_windows1.0.jpg
[image5]: ./output_images/sliding_windows1.2.jpg
[image6]: ./output_images/sliding_windows1.4.jpg
[image7]: ./output_images/sliding_windows1.6.jpg
[image8]: ./output_images/sliding_windows1.8.jpg
[image9]: ./output_images/sliding_windows2.0.jpg
[image10]: ./output_images/test_outputs.jpg
[image11]: ./output_images/bboxes_and_heat.jpg
[image12]: ./output_images/labels_map.jpg
[image13]: ./output_images/output_boxes.jpg

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the code snippet `src/hog.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car image

![alt text][image2]

Not car image

![alt text][image3]

I tried various combinations of parameters (trial and error method) and got optimum results using `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. All 3 channels were used to extract the features as that way we can make use of all the information available to us. `YCrCb` color space was also decided after some trial and error with different color spaces. I noticed that `YCrCb` helps detect white cars better.

I trained a linear SVM using the `hinge` loss function. The features were extracted and then shuffled (in order to avoid overfitting). The dataset was split into training and testing data with a 20% split. The model is saved as a pickle file. This way we can simply load the pickle file for our next run, thus avoiding retraining everytime we want to run our code. The script automatically looks if a pickle file is available in the `src` directory. If not, it trains the model from scratch and saves the pickle file in the `src` directory.

The code for this is available in `src/SVM_classifier.py` file.

### Sliding Window Search

The sliding window code can be found in `src/sliding_window.py` file. I use only the lower half of the image for extracting HOG features to improve speed and accuracy. The cars are only present in the lower half of the images and not in the sky thus we can avoid running our sliding windows in that part. I used a `ystart = 400` and `ystop = 656`.

The extracted HOG features are then subsampled for each sliding window. The `cell_per_block` parameter determines the overlap of each window in terms of cell distance. I tried a lot of different cell sizes and found that using a combination of multiple cell sizes worked the best in my case. Thus I use `scale = 1.0, 1.2, 1.4, 1.6, 1.8 and 2.0` and a `threshold = 5`.

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

Ultimately I searched on six scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image10]
---

### Video Implementation

Here's a [link to my video result](https://www.youtube.com/watch?v=Qo--y7DqCF4)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image11]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image12]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image13]



---

### Discussion

One of the major challenges is to getting the false positives to zero. Here, I used heatmaps and thresholding. Although, this particular input video had very less false positives, a different video might have more. It would be interesting to see how the current pipeline performs on a different video.

Secondly, the hog and sliding window method is slow and resource consuming. The pipeline should be improved and made faster in order to detect vehicles in real-time. Also, as seen in the video, it does fail at times specially with shadows.

I think using advanced deep learning techniques would improve the detection significantly.
