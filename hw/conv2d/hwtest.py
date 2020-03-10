import numpy as np

# Implementation (Grayscale)
def conv2d_grayscale(img, ker):
    """
    Convolve a 2D single-channel image with a 2D single-channel kernel
    using the sliding-window definition.

    img - image (height x width)
    ker - kernel (height x width)
    """

    # Start by decomposing the `.shape`s of the inputs
    # into variables for easier use.
    #
    # See slide 85 for a reference.
    #
    # Note: be very careful with the order of the dimensions!
    #       We used width x height in the lecture, but numpy uses height x width!
    #
    # Here is the numpy documentation of `.shape`:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
    # !!! YOUR CODE HERE
    j, i = ker.shape
    m, n = img.shape
    # !!! ==============

    # Now compute the feature map size. See slide 85 for a reference.
    #
    # You should end up with two numbers (width and height)
    # that are a little smaller than those of the image.
    #
    # !!! YOUR CODE HERE
    Nx = n - i + 1
    Ny = m - j + 1
    # !!! ==============

    # Define an empty numpy array of the right size and type for the feature map.
    #
    # There are many numpy functions that could work here, so practice your google-fu
    # to find the most convenient one.
    #
    # You will need to specify the type (i.e. dtype) of the array for it to work properly.
    # Numpy will throw a helpful error if you get it wrong, but make sure you understand
    # why the type is what it is. Hint: think about the format of our image data
    #
    # !!! YOUR CODE HERE
    feature_map = np.empty([Ny, Nx], dtype = float)
    # !!! ==============

    # Now we just need to iterate over the possible kernel locations,
    # and compute the convolution output at each of them.
    #
    #
    # Check slides 76 and later for a computation of the range of the valid x and y values.
    #
    # The iteration part shouldn't do anything fancy at all, just two lines of pure python.
    #
    #
    # For computing the convolution, you'll need to cut out parts of the image.
    # If you are lost on how to do that, check out this numpy documentation page:
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    #
    # As a reminder, for convolutions, _each output element_ is defined as
    # a _sum_ of the _element-wise product_ of
    # the kernel and the corresponding part of the image.
    #
    # Google the necessary numpy functions!
    # Our solution only uses one in addition to arithmetic operators.
    #
    #
    # Store the result in the feature_map array.
    #
    # !!! YOUR CODE HERE
    for x in range(Nx):
        for y in range(Ny):
            feature_map[y, x] = np.sum(np.multiply(ker, img[y:y+j,x:x+i]))
    # !!! ==============

    return feature_map
