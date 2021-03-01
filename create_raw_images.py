#
# Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import argparse
import numpy as np
import os

def generate_random_snpe_raw(raw_filepath, dim_x, dim_y):
    img = np.random.rand(dim_x, dim_y, 3)
    img_out = img[..., ::-1]
    img_out = img_out.astype(np.float32)
    fid = open(raw_filepath, 'wb')
    img_out.tofile(fid)

def main():
    parser = argparse.ArgumentParser(description='Generate raw images for snpe (BGR format).')
    parser.add_argument('-x', '--x_dim', type=int,
                        help='width of the image to generate.', required=True)
    parser.add_argument('-y', '--y_dim', type=int,
                        help='height of the image to generate.', required=True)
    parser.add_argument('-o', '--output_image',
                        help='name/path of the output image', required=True)

    args = parser.parse_args()

    print(args)
    generate_random_snpe_raw(args.output_image, args.x_dim, args.y_dim)

if __name__ == '__main__':
    main()