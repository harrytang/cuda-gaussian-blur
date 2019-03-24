# /**
# * @author Gia Duy DUONG <giaduy.duong@student.lut.fi>
# * @project Gaussian Blur - GPGPU Computing assignment
# * @copyright Copyright (c) 2019 Gia Duy DUONG
# */

# ########################### #
# import cuda & other modules #
# ########################### #
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
import numpy as np
import math
import sys
import timeit
from PIL import Image

# ########## #
# begin time #
# ########## #
time_started = timeit.default_timer()

# ############### #
# check arguments #
# ############### #
try:
    input_image = str(sys.argv[1])
    output_image = str(sys.argv[2])
except IndexError:
    sys.exit("No input/output image")


# ################################################# #
# load image in to array and extract color channels #
# ################################################# #
try:
    img = Image.open(input_image)
    input_array = np.array(img)
    red_channel = input_array[:, :, 0].copy()
    green_channel = input_array[:, :, 1].copy()
    blue_channel = input_array[:, :, 2].copy()
except FileNotFoundError:
    sys.exit("Cannot load image file")


# ######################################## #
# generate gaussian kernel (size of N * N) #
# ######################################## #
sigma = 2  # standard deviation of the distribution
kernel_width = int(3 * sigma)
if kernel_width % 2 == 0:
    kernel_width = kernel_width - 1  # make sure kernel width only sth 3,5,7 etc

# create empty matrix for the gaussian kernel #
kernel_matrix = np.empty((kernel_width, kernel_width), np.float32)
kernel_half_width = kernel_width // 2
for i in range(-kernel_half_width, kernel_half_width + 1):
    for j in range(-kernel_half_width, kernel_half_width + 1):
        kernel_matrix[i + kernel_half_width][j + kernel_half_width] = (
                np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                / (2 * np.pi * sigma ** 2)
        )
gaussian_kernel = kernel_matrix / kernel_matrix.sum()


# #################################################################### #
# calculate the CUDA threats/blocks/gird base on width/height of image
# #################################################################### #
height, width = input_array.shape[:2]
dim_block = 32
dim_grid_x = math.ceil(width / dim_block)
dim_grid_y = math.ceil(height / dim_block)

# load CUDA code
mod = compiler.SourceModule(open('gaussian_blur.cu').read())
apply_filter = mod.get_function('applyFilter')

# ##################
# apply the  filter
# ##################
for channel in (red_channel, green_channel, blue_channel):
    apply_filter(
        drv.In(channel),
        drv.Out(channel),
        np.uint32(width),
        np.uint32(height),
        drv.In(gaussian_kernel),
        np.uint32(kernel_width),
        block=(dim_block, dim_block, 1),
        grid=(dim_grid_x, dim_grid_y)
    )

# ####################################################################### #
# create the output array with the same shape and type as the input array #
# ####################################################################### #
output_array = np.empty_like(input_array)
output_array[:, :, 0] = red_channel
output_array[:, :, 1] = green_channel
output_array[:, :, 2] = blue_channel

# save result image
Image.fromarray(output_array).save(output_image)

# end time
time_ended = timeit.default_timer()

# display total time
print('Total processing time: ', time_ended - time_started, 's')
