import string
import cv2
import os
import sys
sys.path.append('Preproceccing/EAST')

from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from tqdm import tqdm, trange
#from utils import CTCLabelConverter, AttnLabelConverter
#from dataset import RawDataset, AlignCollate
#from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Preproceccing:
    def __init__(
        self,
        for_learning_output_dirpath,
        book_img_dirpath=None,
        craft_result_dirpath=None,
        segmented_img_dirpath=None,
        char_classification_model_dirpath=None
        ):
        """
        Initialize Preproceccing class object 
        """
        self.for_learning_output_dirpath = for_learning_output_dirpath
        self.book_img_dirpath = book_img_dirpath
        self.craft_result_dirpath = craft_result_dirpath
        self.segmented_img_dirpath = segmented_img_dirpath


    def _saveSegmentedTextRegions(self):
        """ 
        Detect and segment text regions from book cover images, and then save them.
        """
        print(self.craft_result_dirpath)
        if self.craft_result_dirpath is None:
            raise NameError('"self.craft_result_dirpath" is not defined')
        elif self.segmented_img_dirpath is None:
            raise NameError('"self.segmented_img_dirpath" is not defined')

        self.asins = [None] * len(self.book_img_filepaths)
        from segmentation import Segmentation

        if not os.path.isdir(self.segmented_img_dirpath):
            os.makedirs(self.segmented_img_dirpath)

        for i, filepath in enumerate(tqdm(self.book_img_filepaths)):
            filename = os.path.basename(filepath)
            stem = os.path.splitext(filename)[0]
            self.asins[i] = stem
            output_dirpath = os.path.join(self.segmented_img_dirpath, stem)
            if os.path.isdir(output_dirpath):
                continue
            txt_filepath = os.path.join(self.craft_result_dirpath, stem + '.txt')
            if not os.path.isfile(txt_filepath):
                continue
            x = Segmentation(filepath, txt_filepath)
            x.detectAndSegmentChars(output_for_learning_dirpath=self.for_learning_output_dirpath,output_for_charRecognition_dirpath=self.segmented_img_dirpath)


    def _classifyCharacters(opt):
        """ vocab / character number configuration """

        #character="0123456789abcdefghijklmnopqrstuvwxyz"
        character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

        """ model configuration """
        
        converter = AttnLabelConverter(character)

        num_class = len(converter.character)

        rgb = False

        if rgb:
            input_channel = 3
        
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)

        # predict
        model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)


                log = open(f'./log_demo_result.txt', 'w')
                dashed_line = '-' * 80
                head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
                
                print(f'{dashed_line}\n{head}\n{dashed_line}')
                #log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    pred = pred[0]

                    print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                    log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

                log.close()

    def _storeBookImageFilepaths(self):
        """ 
        Store filepaths of book cover images.
        """
        if self.book_img_dirpath is None:
            raise NameError('"self.book_img_dirpath" is not defined')

        if os.path.isdir(self.book_img_dirpath):
            from tqdm import trange
        
            self.book_img_filepaths = []

            for filename in os.listdir(self.book_img_dirpath):
                stem, ext = os.path.splitext(filename)
                self.book_img_filepaths.append(os.path.join(self.book_img_dirpath, filename))

            self.book_img_filepaths.sort()

        else:
            raise ValueError('"self.book_img_dirpath" must be a directory path including images')
    
    def run(self):
        """
        Main function of Preproceccing Class
        """

        print('[Store book cover images\' filepath from "{}"]'.format(self.book_img_dirpath))
        self._storeBookImageFilepaths()
        print('.. #images: {}'.format(len(self.book_img_filepaths)))

        print('\n[Detect and segment candidates of characters from the input images]')
        self._saveSegmentedTextRegions()

        #print('\n[Classify loaded character (segmented image by craft )\' alpahbet]')
        #self._classifyCharacters()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'for_learning_output_dirpath', 
        type=str,
        help='output directory path for learning'
    )
    parser.add_argument(
        '--book_img_dirpath',
        type=str,
        default=None,
        help='path of directory including test images (default: None)'
    )
    parser.add_argument(
        '--craft_result_dirpath',
        type=str,
        default=None,
        help=('path of directory including .txt files about text region detection based on CRAFT'
              + ' (default: None)')
    )
    parser.add_argument(
        '--segmented_img_dirpath',
        type=str,
        default=None,
        help='path of directory including segmented character images (default: None)'
    )

    args = parser.parse_args()

    x = Preproceccing(
        args.for_learning_output_dirpath,
        book_img_dirpath=args.book_img_dirpath,
        craft_result_dirpath=args.craft_result_dirpath,
        segmented_img_dirpath=args.segmented_img_dirpath
    )
    x.run()
