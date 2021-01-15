import itertools
import os
import random
import six
import numpy as np
import cv2
import keras

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")

    def tqdm(iter):
        return iter


DATA_LOADER_SEED = 0

random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


class DataLoaderError(Exception):
    pass


def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]

    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
           os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(segs_path, dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0}"
                                      " already exists and is ambiguous to"
                                      " resolve with path {1}."
                                      " Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))

            segmentation_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path,
                                segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))

    return return_value


def get_image_array(image_input,
                    width, height,
                    imgNorm="sub_mean", ordering='channels_first'):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels


def verify_segmentation_dataset(images_path, segs_path,
                                n_classes, show_all_errors=False):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        if not len(img_seg_pairs):
            print("Couldn't load any data from images_path: "
                  "{0} and segmentations path: {1}"
                  .format(images_path, segs_path))
            return False

        return_value = True
        for im_fn, seg_fn in tqdm(img_seg_pairs):
            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            # Check dimensions match
            if not img.shape == seg.shape:
                return_value = False
                print("The size of image {0} and its segmentation {1} "
                      "doesn't match (possibly the files are corrupt)."
                      .format(im_fn, seg_fn))
                if not show_all_errors:
                    break
            else:
                max_pixel_value = np.max(seg[:, :, 0])
                if max_pixel_value >= n_classes:
                    return_value = False
                    print("The pixel values of the segmentation image {0} "
                          "violating range [0, {1}]. "
                          "Found maximum pixel value {2}"
                          .format(seg_fn, str(n_classes - 1), max_pixel_value))
                    if not show_all_errors:
                        break
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False


def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width, weights=(1.,1.)):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        W = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)

            X.append(get_image_array(im, input_width,
                                     input_height, ordering="channels_last"))
            mask = get_segmentation_array(
                seg, n_classes, output_width, output_height, no_reshape=False)
            Y.append(mask)
            weights_arr = mask[:,1].copy()
            for i, mi in enumerate(mask[:,1]):
                if mi < 0.5:
                  weights_arr[i] = weights[0]
                else:
                  weights_arr[i] = weights[1]
            W.append(weights_arr)
            #print(weights_arr)
            #print(mask[:,1])
        yield np.array(X), np.array(Y), np.array(W)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                images_path, segs_path, batch_size, input_height, input_width,
                output_height, output_width):#, weights=(1.,1.)):
        'Initialization'
        self.images_path = images_path
        self.segs_path = segs_path
        self.batch_size = batch_size
        self.n_classes = 2
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        #self.weights = weights

        self.img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        self.indexes = np.arange(len(self.img_seg_pairs))
        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
        rs.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        l = int(len(self.img_seg_pairs) / self.batch_size)
        #if l*self.batch_size < len(self.img_seg_pairs):
        #    l += 1
        return l
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(indexes)
        # Find list of IDs
        img_seg_pairs_temp = [self.img_seg_pairs[k] for k in indexes]
        # Generate data
        #if self.weights is not None:
        #    X, Y, W = self.__data_generation(img_seg_pairs_temp)
        #    return X, Y, W
        #else:
        X, Y = self.__data_generation(img_seg_pairs_temp)

        return X, Y
        #print(X,Y,W)
           

    def on_epoch_end(self):
        pass

    def __data_generation(self, img_seg_pairs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #print(len(img_seg_pairs))
        X = np.empty((len(img_seg_pairs), self.input_height, self.input_width, 3))
        Y = np.empty((len(img_seg_pairs), self.output_height*self.output_width, self.n_classes), dtype=int)
        #W = np.empty((len(img_seg_pairs), self.output_height*self.output_height))
        # Generate data
        for i, pair in enumerate(img_seg_pairs):

            im = cv2.imread(pair[0], 1)
            seg = cv2.imread(pair[1], 1)

            mask = get_segmentation_array(
                seg, self.n_classes, self.output_width, self.output_height, no_reshape=False)
                        # Store sample
            X[i] = get_image_array(im, self.input_width,
                                     self.input_height, ordering="channels_last")
            # Store class
            Y[i] = mask
            """
            if self.weights is not None:
                weights_arr = mask[:,1].copy()
                for j, mi in enumerate(mask[:,1]):
                    if mi < 0.5:
                      weights_arr[j] = self.weights[0]
                    else:
                      weights_arr[j] = self.weights[1]
                W[i] = weights_arr
                #print(weights_arr)
                #print(mask[:,1])
                return X, Y, W
            """
        return X, Y