import os
import os.path as osp
import logging
import argparse
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
import time

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Only use GPU for testing
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs((path for key, path in opt['path'].items()
                 if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    dataset_opt = opt['datasets']['test']
    test_set = create_dataset(dataset_opt['test_fIQA'], 'valid')
    test_loader = create_dataloader(test_set, dataset_opt)

    model = create_model(opt)
    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    print(f"*** test_names: {dataset_opt['test_fIQA']['name']}")
    
    logger.info('\nTesting [{:s}] ...'.format(dataset_opt['test_fIQA']['name']))
    dataset_dir = osp.join(opt['path']['results_root'], dataset_opt['test_fIQA']['name'] + '_' + curr_time)
    logger.info(f'\nTesting dataset_dir {dataset_dir}')
    util.mkdir(dataset_dir)
    index = 0
    txt_line_list = []
    txt_line = 'image name' + ',' + 'Comprehensive' + ',' + 'Noise' + ',' + 'Blur' + ',' + 'Color' + ',' + 'Contrast' + '\n'
    txt_line_list.append(txt_line)
    start_time = time.time()
    for data_pair in test_loader:
        model.feed_data(data_pair, Train=False)
        model.test()
        noise_pre, blur_pre, color_pre, contrast_pre = model.get_current_score()
        noise_score = float(noise_pre.numpy())
        blur_score = float(blur_pre.numpy())
        color_score = float(color_pre.numpy())
        contrast_score = float(contrast_pre.numpy())
        Comprehensive_score = (noise_score + blur_score + color_score + contrast_score) / 4
        Dist_name = data_pair['name'][0].split('/')[-1]
        txt_line = Dist_name + ',' + str(Comprehensive_score) + ',' + str(noise_score) + ',' + str(blur_score)+ ',' + str(color_score)+ ',' + str(contrast_score) + '\n'
        txt_line_list.append(txt_line)
        index += 1
        print(f"Process No.{index} Image Comprehensive_score: {str(Comprehensive_score)} noise_score: {str(noise_score)} blur_score: {str(blur_score)} color_score: {str(color_score)} contrast_score: {str(contrast_score)}")
        logger.info('Process No.{} Image Comprehensive_score: {} noise_score: {} blur_score: {} color_score: {} contrast_score: {}'.format(index, str(Comprehensive_score), str(noise_score), str(blur_score), str(color_score), str(contrast_score)))

    end_time = time.time()
    time_diff = end_time - start_time
    per_image_time = time_diff / index
                
    with open(os.path.join(dataset_dir, "output.txt"), 'a') as f:
        for line in txt_line_list:
            f.write(line)
            
    with open(os.path.join(dataset_dir, "readme.txt"), 'a') as f:
        f.write("runtime per image [s] : " + str(per_image_time)+"\n")
        f.write("CPU [1] / GPU [0] : " + str(1)+"\n")
        f.write("Extra Data [1] / No Extra Data [0] : " + str(0)+"\n")
        f.write("Other description : Solution based on PIPAL of Gu et al. ECCV 2020. We have a PyTorch implementation," + "\n")
        f.write("and report single core CPU runtime. The method was trained on PIPAL and face IQA dataset" + "\n")
        

if __name__ == '__main__':
    main()