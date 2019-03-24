CUDA Gaussian Blur
==================

Introduction
------------
CUDA is a parallel computing platform and application programming interface model created by Nvidia. 
In this report, I will show the Gaussian Blur algorithm along with the implementation in PyCUDA. 

Prerequisites
-------------
To complete this implementation, you will need a computer with a CUDA-capable GPU. You also need to install and
configure some software, more info please see https://documen.tician.de/pycuda/ and https://developer.nvidia.com/cuda-zone

Implementation
--------------
The Gaussian Blur algorithm is easy to implement, it uses a convolution kernel. The algorithm can be slow as it's
processing time is dependent on the size of the image and the size of the kernel.

Step 1 - Load the input image, extract all the color channels (red, green, blue) of the image.

Step 2 - Select the size of the kernel, then use the formula of a Gaussian function to generate the matrix kernel. In
this sample source code, the size of the kernel is 5x5

Step 3 - Convolution of image with kernel. If the image has 3 color channels, we process all the individual color
channel separately by multiple the pixel value of every pixel corresponding to its location in the convolution kernel.
Save the result in the output arrays.

Step 4 - Merge all the output arrays (red, green, blue) and save as an output result image which is already blurred.

Usage
-----

`python main.py test.tif result.tif`

Conclusion
----------
