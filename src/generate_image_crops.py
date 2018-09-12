import numpy as np
from PIL import Image


def generate_crops(image, nr_crops, crop_size):
    image_size = image.size
    f = open("images/labels.txt", "w")
    for i in range(nr_crops):
        new_x = np.random.uniform(0, image_size[0] - crop_size[0] - 1)
        new_y = np.random.uniform(0, image_size[1] - crop_size[1] - 1)
        label = (new_x + crop_size[0] * 0.5, new_y + crop_size[1] * 0.5)
        print(label, file=f)
        new_crop_shape = (new_x, new_y, new_x + crop_size[0], new_y + crop_size[1])
        new_crop = image.crop(new_crop_shape)
        new_crop.save("images/{}.png".format(i))
        if i % 10 == 0:
            print("image {} done".format(i))

    f.close()


if __name__ == "__main__":
    image_name = "sistine-chapel.jpg"
    nr_crops = 1000
    crop_size = (2000, 2000)
    image = Image.open(image_name)
    generate_crops(image, nr_crops, crop_size)
