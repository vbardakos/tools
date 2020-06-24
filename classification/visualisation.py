import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg


class ClassSample(object):
    """
    Creates, organises and plots a random sample of images per class.

    :@class from_directories : input a series of image directories
    :@class from_images : input a series of images
    :method show : Shows figure
    """

    def __init__(self, samples: np.ndarray, size: int, path: bool):
        """
        Use from_directories & from_images class methods instead
        """
        self.data = samples
        self.size = size
        self.path = path

    @classmethod
    def from_directories(cls, *dirs: str, sample_size: int, root_path: str = ""):
        """
        Gets a random sample from given class directories
        Example :
            >> path = 'tmp/train'
            >> os.listdir(path)  # check classes' path
            ['cat', 'dog']

            >> # show a 2x4 figure with 4 cat and 4 dog images
            >> ClassSample('dog', 'cat', sample_size=4, root_path=path).show()

        :param dirs: series of class directories
        :param sample_size: number of random images per class to generate
        :param root_path: common path of directories used (optional)
        :return: cls.__init__
        """
        dirs = np.char.array(dirs).flatten()
        if root_path:
            args = np.char.add(root_path + os.path.sep, dirs)
        # get sample
        files = np.empty(0)
        for path in dirs:
            temp = np.char.array(np.char.add(path + os.path.sep,
                                             os.listdir(path)))
            files = np.append(files, np.random.choice(temp, sample_size))

        return cls(files, sample_size, path=True)

    @classmethod
    def from_images(cls, *imgs, sample_size: int, labels: list = list()):
        """
        Gets a random sample from given class directories

        :param imgs: lists of class images
        :param sample_size: number of random images per class to generate
        :param labels: if all classes are in a single list, use labels to
                       classify them. Note: len(labels) and len(images) should
                       be equal.
        :return: cls.__init__
        """
        # TODO : Run & Debug if needed
        args = np.array(imgs)
        images = np.empty(0)
        if labels:
            args = args.flatten()
            labels = np.array(labels)
            if args.size == labels.size:
                idx = np.max(labels)
                for label in range(idx + 1):
                    mask = labels == label
                    images = np.append(images, np.random.choice(args[mask], sample_size))
            else:
                raise Exception("Images and labels should have the same size")
        else:
            for img in args:
                np.append(images, np.random.choice(img, sample_size))

        return cls(images, sample_size, path=False)

    def show(self, spacing: int = 4, class_cols: int = 0):
        """
        Plots given images or directories

        :param spacing: spacing between images (optional)
        :param class_cols: number of columns (optional)
        """
        total_cols, total_rows = self.__img_box__(class_cols)

        print(total_cols, total_rows, self.size)

        figure = plt.gcf()
        # set figure with subplots
        figure.set_size_inches(total_cols * spacing, total_rows * spacing)
        for idx, img in enumerate(self.data):
            subplot = plt.subplot(total_rows, total_cols, idx + 1)
            subplot.axis('Off')
            if self.path:
                img = mimg.imread(img)
            plt.imshow(img)

        plt.show()

    def __img_box__(self, cols):
        """
        Utility method to organise images in boxes before plotting
        :return: (total_columns, total_rows)
        """
        if cols:
            if self.size < cols or self.size % cols:
                raise Exception("Invalid class cols param")
            else:
                rows = self.size / cols
        elif self.size < 5:
            cols = self.size
            rows = 1
        else:
            for c in range(self.size, 3, -1):
                if not self.size % c:
                    cols = c
                    rows = self.size / c
        rows = np.int((self.data.size / self.size) * rows)

        return cols, rows


if __name__ == '__main__':
    # get images
    image_path = 'directory/to/train'
    vis = ClassSample.from_directories('cat', 'dog', sample_size=6, root_path=image_path)
    # visualise
    vis.show(class_cols=3)
