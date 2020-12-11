import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

import csv
import os

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _textRecognition(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
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

    csv_filename = os.path.join(opt.output_dirpath,opt.csv_file_name)
    Header = ["bookID","prediction"]

    with open(csv_filename,'w') as f:
        writer = csv.DictWriter(f,fieldnames=Header)
        writer.writeheader()

        model.eval()
        with torch.no_grad():
            pbar = tqdm(len(demo_loader),total=len(demo_loader))  
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


                #log = open(f'./text_information.csv', 'a')
                dashed_line = '-' * 80
                head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
                
                #print(f'{dashed_line}\n{head}\n{dashed_line}')

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    
                    try:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                    except:
                        delete_title_region_and_cover(opt.output_dirpath,img_name)
                        continue
                    
                    #confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    #pred = pred[0]

                    if confidence_score < 0.6 :
                        pred = "Unreadble"
                        delete_title_region_and_cover(opt.output_dirpath,img_name)
                        continue
    
                    # extract the name part of the image
                    filename = os.path.basename(img_name)
                    
                    pbar.set_description("filename:{},pred:{}".format(filename,pred))
                    #print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                    
                    writer.writerow({"bookID":filename,"prediction":pred})
                
                pbar.update(1)
            pbar.close()

def delete_title_region_and_cover(output_dirpath,img_for_recognition_filepath):

    if os.path.isfile(img_for_recognition_filepath):
        paths = os.listdir(output_dirpath)
        filename = os.path.basename(img_for_recognition_filepath)

        for path in paths:
            target_dirpath = os.path.join(output_dirpath,path)
            target_filepath = os.path.join(target_dirpath,filename)

            if os.path.isfile(target_filepath):
                os.remove(target_filepath)
                continue
            else:
                continue
    else:
        raise NameError("such file is not exist.")
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--book_img_dirpath', required=True, help='path to image_folder which contains target images')
    parser.add_argument('--csv_file_name', required=True, help='name of the csv file text info saved in')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--output_dirpath', required=True, help="path to output dirpath")

    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    _textRecognition(opt)
