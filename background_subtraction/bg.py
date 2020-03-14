import cv2 as cv
import os, ntpath


class StaticSubtractor:

    def __init__(self, static_image):
        self.static_image = cv.imread(static_image)

    def apply(self, frame):
        return cv.absdiff(frame, self.static_image)


class ImageWriter:

    def __init__(self, out_file_loc):
        self.out_file_loc = out_file_loc
        self.counter = 0

    def write(self, frame):
        cv.imwrite(f"{self.out_file_loc}/{self.counter}.bmp", frame)
        self.counter += 1


def colour_threshold(frame):
    # lower_bound = (120, 120, 120)
    # upper_bound = (255, 255, 255)

    lower_bound = (0,0,0)
    upper_bound = (150,150,150)

    mask = cv.inRange(frame, lower_bound, upper_bound, cv.THRESH_TOZERO_INV)
    return cv.bitwise_and(frame, frame, mask=mask)


def subtract_background_rgb(in_file_loc: str, out_file_loc: str, save_as_images=False):
    augment_video(in_file_loc, out_file_loc, colour_threshold, save_as_images=save_as_images)


def subtract_background_mog(in_file_loc: str, out_file_loc: str, save_as_images=False,  var_threshold=16, detect_shadows=False):
    back_sub = cv.createBackgroundSubtractorMOG2(varThreshold=var_threshold, detectShadows=detect_shadows)

    back_sub_apply = lambda frame: cv.bitwise_and(frame, frame, mask=back_sub.apply(frame))
    augment_video(in_file_loc, out_file_loc, back_sub_apply, save_as_images=save_as_images)


def subtract_background_static(in_file_loc: str, out_file_loc: str, static_image_loc: str, save_as_images=False):
    static_subtractor = StaticSubtractor(static_image=static_image_loc)
    augment_video(in_file_loc, out_file_loc, static_subtractor.apply, save_as_images=save_as_images)


def augment_video(in_file_loc: str, out_file_dir: str, preprocess_func, save_as_images = False):

    if not os.path.exists(out_file_dir):
        os.mkdir(out_file_dir)

    if not os.path.isdir(out_file_dir):
        raise NotADirectoryError('out_file_dir must be a directory! (and different from your input directory)')

    capture = cv.VideoCapture(in_file_loc)

    if not capture.isOpened():
        raise FileNotFoundError('Unable to open: ' + in_file_loc)

    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv.CAP_PROP_FPS))

    if save_as_images:
        out = ImageWriter(out_file_dir)
    else:
        out = cv.VideoWriter(os.path.join(out_file_dir, ntpath.basename(in_file_loc)), cv.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        augmented_frame = preprocess_func(frame)

        cv.imshow('Frame', frame)
        cv.imshow('Augmented', augmented_frame)

        cv.waitKey(30)

        #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        out.write(augmented_frame)

    capture.release()

    if not save_as_images:
        out.release()

#subtract_background_static(in_file_loc=os.getcwd() + '/data/videos/0.avi', out_file_loc=os.getcwd() + '/data/bs_videos/withstatic', static_image_loc=os.getcwd() + '/data/static_background_images/0.bmp')
#subtract_background_mog(in_file_loc=os.getcwd() + '/data/videos/0.avi', out_file_loc=os.getcwd() + '/data/bs_videos/A', var_threshold=12)

