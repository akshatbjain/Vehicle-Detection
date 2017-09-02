import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os.path
import pickle
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

import data_exploration as de
import SVM_classifier as svm
import sliding_window as sw
import hog
import remove_false_positives as rfp
import get_bounding_boxes as gbb

# Hyperparameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()

def explore_data():
    cars_path = "../dataset/vehicles/*/*.png"
    notcars_path = "../dataset/non-vehicles/*/*.png"
    cars_list, notcars_list = de.read_image_list(cars_path, notcars_path)
    de.print_data_info(cars_list, notcars_list)
    #de.show_random_dataset_images(cars_list, notcars_list)

    return cars_list, notcars_list

def pipeline(image):
    draw_image = np.copy(image)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # We extracted training data from .png images (scaled 0 to 1 by mpimg) and the
    # image we are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255

    bbox = gbb.find_cars(image, 400, 656, 1.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    bbox += gbb.find_cars(image, 400, 656, 1.33, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    bbox += gbb.find_cars(image, 400, 656, 1.67, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    bbox += gbb.find_cars(image, 400, 656, 2.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # Add heat to each box in box list
    # heat = add_heat(heat,hot_windows)
    heat = rfp.add_heat(heat, bbox)

    # Apply threshold to help remove false positives
    heat = rfp.apply_threshold(heat,2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = rfp.draw_labeled_bboxes(draw_image, labels)

    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    # plt.subplot(122)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # fig.tight_layout()
    # plt.show()

    return draw_img

if __name__ == "__main__":
    cars_list, notcars_list = explore_data()

    if (os.path.isfile('model.p')):
        with open('model.p', 'rb') as f:
            model = pickle.load(f)
        svc = model['svc']
        X_scaler = model['X_scaler']
    else:
        svc, X_scaler = svm.run_classifer(cars_list, notcars_list, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    input_file = '../project_video.mp4'
    output_file = '../project_output.mp4'

    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(pipeline)
    annotated_video.write_videofile(output_file, audio=False)

    # video = cv2.VideoCapture('../project_video.mp4')
    #
    # while(video.isOpened()):
    #     ret, frame = video.read()
    #
    #     if(ret):
    #         output = pipeline(frame)
    #         out.write(np.uint8(output))
    #         # cv2.imshow("output", output)
    #         # cv2.waitKey(1)
    #     else:
    #         break
    #
    # video.release()
    # cv2.destroyAllWindows()
