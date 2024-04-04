#unzip the gz file in the folder
import gzip
import shutil
import os
import sys
import struct
import numpy as np
import cv2

def extract_gz(gz_file, output_file):
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
#extract the emnist-letters-train-images-idx3-ubyte.gz file
extract_gz('emnist-letters-train-images-idx3-ubyte.gz', 'emnist-letters-train-images-idx3-ubyte')
#extract the emnist-letters-train-labels-idx1-ubyte.gz file
extract_gz('emnist-letters-train-labels-idx1-ubyte.gz', 'emnist-letters-train-labels-idx1-ubyte')
#extract the emnist-letters-test-images-idx3-ubyte.gz file
extract_gz('emnist-letters-test-images-idx3-ubyte.gz', 'emnist-letters-test-images-idx3-ubyte')
#extract the emnist-letters-test-labels-idx1-ubyte.gz file
extract_gz('emnist-letters-test-labels-idx1-ubyte.gz', 'emnist-letters-test-labels-idx1-ubyte')