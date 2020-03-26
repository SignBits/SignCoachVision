import cv2 as cv
from preprocessing.util import augment_video
import os, ntpath


class StaticSubtractor:

    def __init__(self, static_image):
        self.static_image = cv.imread(static_image)

    def apply(self, frame):
        return cv.absdiff(frame, self.static_image)


def colour_threshold(frame):

    lower_bound = (0,0,0)
    upper_bound = (150,150,150)

    mask = cv.inRange(frame, lower_bound, upper_bound, cv.THRESH_TOZERO_INV)
    return cv.bitwise_and(frame, frame, mask=mask)


def subtract_background_rgb(in_file_loc: str, out_file_loc: str, save_as_images=False):
    augment_video(
        in_file_loc,
        out_file_loc,
        colour_threshold,
        save_as_images=save_as_images
    )


def subtract_background_mog(in_file_loc: str, out_file_loc: str, save_as_images=False,  var_threshold=16, detect_shadows=False):
    back_sub = cv.createBackgroundSubtractorMOG2(varThreshold=var_threshold, detectShadows=detect_shadows)

    augment_video(
        in_file_loc,
        out_file_loc,
        lambda frame: cv.bitwise_and(frame, frame, mask=back_sub.apply(frame)),
        save_as_images=save_as_images
    )


def subtract_background_static(in_file_loc: str, out_file_loc: str, static_image_loc: str, save_as_images=False):
    static_subtractor = StaticSubtractor(static_image=static_image_loc)
    augment_video(
        in_file_loc,
        out_file_loc,
        static_subtractor.apply,
        save_as_images=save_as_images
    )

#subtract_background_static(in_file_loc=os.getcwd() + '/data/videos/0.avi', out_file_loc=os.getcwd() + '/data/bs_videos/withstatic', static_image_loc=os.getcwd() + '/data/static_background_images/0.bmp')
#subtract_background_mog(in_file_loc=os.getcwd() + '/data/videos/0.avi', out_file_loc=os.getcwd() + '/data/bs_videos/A', var_threshold=12)

