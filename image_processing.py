





if __name__ == '__main__':
    from argparse import ArgumentParser
    import os
    
    parser = ArgumentParser()
    parser.add_argument(
        'img_path',
        type=str,
        help='test image filepath or path of directory including test images'
    )
    parser.add_argument(
        'txt_path',
        type=str,
        help=('path of .txt file in which is written the coordinates of detected text region'
              + 'bounding boxes or path of directory which includes them')
    )
    parser.add_argument(
        'output_for_learning_dirpath',
        type=str,
        help='output directory path of the images for machine learning(mask and inpainting)'
    )
    parser.add_argument(
        'output_for_charRecognition_dirpath',
        type=str,
        help='output directory path of segmented character images'
    )
    args = parser.parse_args()
    
    main(args)
