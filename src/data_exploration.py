import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

def read_image_list(cars_path, notcars_path):
    # images are divided up into vehicles and non-vehicles
    cars_list = glob.glob(cars_path)
    notcars_list = glob.glob(notcars_path)

    return cars_list, notcars_list

# Prints characteristics of the dataset
def print_data_info(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    test_image = cv2.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = test_image.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = type(test_image[0,0,0])

    print('\033[94mYour function returned a count of {} cars and {} non-cars of size: {} and data type: {}\033[0m'.format(data_dict["n_cars"], data_dict["n_notcars"], data_dict["image_shape"], data_dict["data_type"]))

def show_random_dataset_images(cars_list, notcars_list):
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars_list))
    notcar_ind = np.random.randint(0, len(notcars_list))

    # Read in car / not-car images
    car_image = mpimg.imread(cars_list[car_ind])
    notcar_image = mpimg.imread(notcars_list[notcar_ind])

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    # plt.savefig('../output_images/car_not_car.jpg')
    plt.show()
