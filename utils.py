import argparse
import cv2
import torch
import numpy as np


def crop_center(image): # takes image as input
    # crop the center of an image and matching the height with the width of the image
    shape = image.shape[:-1] # stores dimensions of the input image
                             # [:-1] will remove last dimension of the shape, which is the color channel
    max_size_index = np.argmax(shape) # finds the index of the largest dimension of the image shape
    diff1 = abs((shape[0] - shape[1]) // 2) # calculates the amount of padding that needs to be added to the smaller 
                                            # dimension of the image to make it equal to the larger dimension
                                            # this is done by taking the absolute value of the difference between
                                            # the two dimensions and dividing by two
    diff2 = shape[max_size_index] - shape[1 - max_size_index] - diff1 
                                        # this calculates the amlount of padding needed to be added to 
                                        # the other dimension of the image. This is done by subtracting
                                        # the smaller dimensionsion from the larger dimension and substracting
                                        # the amount of padding added in the previous step
    return image[:, diff1: -diff2] if max_size_index == 1 else image[diff1: -diff2, :]
                                        # this will return the cropped image
                                        # if the largerst dimension is the second dimension (i.e width > length)
                                        # the function will crop the image horizontally by taking all rows and columns
                                        # from diff1 to -diff2
                                        # otherwise the largest dimension is the first (height > width)


# the purpose of this function is to return a PyTorch data type either `torch.cude.FloatTensor` or `torch.FloatTensor`, 
# depednding on whethere a CUDA-capable GPU is available or not
def get_dtype():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu' # checks if a CUDA-capable GPU is available. 
                                                         # if it's available it sets the variable `dev` to `'cuda'`
                                                         # otherwise, it will set `dev` to `cpu`
    device = torch.device(dev) # creates a PyTorch device object based on the value of `dev`
    if dev == 'cuda':
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    print(f'Using device {device}') # prints the device being used
    return dtype 

# CUDA stands for Compute Unified Device Architecture and is a parallel computing platform and programming model created by NVIDIA


# takes a video file as input and returns properties including the frame rate (fps), length, width, and height of each frame
# it uses OpenCV (cv2) library to extract these properties
def get_video_properties(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') # splits the OpenCV version number into three integers, separated by '.'

    # get videos properties
    if int(major_ver) < 3: # checks if the major version number is less than 3, meaning older version is being used
                            # will then gather characteristics using OpenCV constants
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    else: # otherwise, the version is 3 or greater, meaning we use the updated OpenCV constants 
        fps = video.get(cv2.CAP_PROP_FPS) 
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height


def str2bool(v): # function to convert a string representation of a boolean value to its corresponding boolean value
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_stickman_line_connection(): # function to return a list of tuples that represent the connections between key points in a stickman figure 
                                    # used in R-CNN (Region-based Convolutional Neural Network) object detection
                                    
    # stic kman line connection with keypoints indices for R-CNN
    line_connection = [
        (7, 9), (7, 5), (10, 8), (8, 6), (6, 5), (15, 13), (13, 11), (11, 12), (12, 14), (14, 16), (5, 11), (12, 6)
    ]
    return line_connection
