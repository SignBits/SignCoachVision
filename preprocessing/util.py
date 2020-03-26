import cv2 as cv
import os, ntpath


class ImageWriter:

    def __init__(self, id, out_file_loc, write_frequency):
        self.out_file_loc = out_file_loc
        self.counter = 0
        self.id = id
        self.write_frequency = write_frequency

    def write(self, frame):
        if self.counter % self.write_frequency == 0:
            cv.imwrite(f"{self.out_file_loc}/{self.id}_{self.counter}.bmp", frame)

        self.counter += 1


def multiple_avi_to_bmp(in_file_dir: str, out_file_dir: str, write_frequency, preprocess_func):
    for subdirs, dirs, files in os.walk(in_file_dir):
        for file in files:
            avi_to_bmp(os.path.join(in_file_dir, file), out_file_dir, write_frequency, preprocess_func)


def avi_to_bmp(in_file_loc: str, out_file_dir: str, write_frequency, preprocess_func):
    augment_video(in_file_loc, out_file_dir, write_frequency, preprocess_func, save_as_images=True)


def augment_video(in_file_loc: str, out_file_dir: str, write_frequency, preprocess_func, fourcc='XVID',
                  save_as_images=False):
    """This function takes a video file 'AVI' by default (change fourcc for different filetype) and augments
        each frame given the preprocessing_function. It then saves out_file_dir as either a single augmented video,
        or if save_as_images = True, saves the images to out_file_dir"""

    if preprocess_func is None:
        preprocess_func = lambda frame: frame

    if not os.path.exists(out_file_dir):
        os.mkdir(out_file_dir)

    if not os.path.isdir(out_file_dir):
        # Must be a directory to prevent ambiguity regarding whether to save as images or video
        raise NotADirectoryError('out_file_dir must be a directory! (and different from your input directory)')

    capture = cv.VideoCapture(in_file_loc)

    if not capture.isOpened():
        raise FileNotFoundError('Unable to open: ' + in_file_loc)

    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv.CAP_PROP_FPS))

    if save_as_images:
        out = ImageWriter(os.path.basename(in_file_loc), out_file_dir, write_frequency)
    else:
        out = cv.VideoWriter(
            os.path.join(out_file_dir, ntpath.basename(in_file_loc)),
            cv.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height)
        )

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        augmented_frame = preprocess_func(frame)

        # cv.imshow('Frame', frame)
        # cv.imshow('Augmented', augmented_frame)
        #
        # cv.waitKey(30)

        # frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        out.write(augmented_frame)

    capture.release()

    if not save_as_images:
        out.release()


def final(in_file_dir, out_file_dir, preprocess_func=None, write_frequency=4):
    for subdirs, dirs, files in os.walk(in_file_dir):
        for directory in dirs:
            multiple_avi_to_bmp(
                in_file_dir=os.path.join(in_file_dir, directory),
                out_file_dir=os.path.join(out_file_dir, directory),
                preprocess_func=preprocess_func,
                write_frequency=write_frequency
            )


final("/Users/mclancy/Documents/sdp/preprocesstest/folder",
      "/Users/mclancy/Documents/sdp/preprocesstest/folderout",
      )
