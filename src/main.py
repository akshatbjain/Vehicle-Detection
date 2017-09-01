import data_exploration as de

cars_path = "../dataset/vehicles/*/*.png"
notcars_path = "../dataset/non-vehicles/*/*.png"
cars_list, notcars_list = de.read_image_list(cars_path, notcars_path)
de.print_data_info(cars_list, notcars_list)
de.show_random_dataset_images(cars_list, notcars_list)
