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
    #### options
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Only use GPU for testing
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    args = parser.parse_args()
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/options/options.py def parse(opt_path, is_train=True)
    opt = option.parse(args.opt, is_train=False)
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/options/options.py def dict_to_nonedict(opt)
    opt = option.dict_to_nonedict(opt)

    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/utils/util.py def mkdirs(paths)
    util.mkdirs((path for key, path in opt['path'].items()
                 if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/utils/util.py def setup_logger
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/options/options.py def dict2str(opt, indent_l=1)
    logger.info(option.dict2str(opt))
    # Create test dataset and dataloader
    dataset_opt = opt['datasets']['test']
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/__init__.py def create_dataset(dataset_opt, mode)
    test_set = create_dataset(dataset_opt['test_fIQA'], 'valid')
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/__init__.py def create_dataloader(dataset, dataset_opt, opt=None, sampler=None)
    test_loader = create_dataloader(test_set, dataset_opt)
    # logger.info('Number of val images in [{:s}]: {:d}'.format(test_name, len(test_set)))
    # Number of val images in [test_PIPAL_Full]: 5800

    # create model and load data
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/__init__.py def create_model(opt)
    model = create_model(opt)
    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    print(f"*** test_names: {dataset_opt['test_fIQA']['name']}")
    
    logger.info('\nTesting [{:s}] ...'.format(dataset_opt['test_fIQA']['name']))
    # dataset_dir = osp.join(opt['path']['results_root'], test_name)
    dataset_dir = osp.join(opt['path']['results_root'], dataset_opt['test_fIQA']['name'] + '_' + curr_time)
    # print(f"****** Testing dataset_dir: {dataset_dir}")
    logger.info(f'\nTesting dataset_dir {dataset_dir}')
    # print(f"test_name: {test_name} dataset_dir: {dataset_dir}")
    # test_name: test_PIPAL_Full 
    # dataset_dir: /public/liuhongzhi/Method/IQA/SEU-IQA/results/LPIPS-Alex_TonPIPAL/test_PIPAL_Full_2024-07-31-20-07-37
    # time.sleep(5)
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/utils/util.py def mkdir(path)
    util.mkdir(dataset_dir)
    index = 0
    txt_line_list = []
    txt_line = 'image name' + ',' + 'Comprehensive' + ',' + 'Noise' + ',' + 'Blur' + ',' + 'Color' + ',' + 'Contrast' + '\n'
    txt_line_list.append(txt_line)
    start_time = time.time()
    for data_pair in test_loader:
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/IQA_model.py def feed_data(self, data, Train=True)
        model.feed_data(data_pair, Train=False)
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/IQA_model.py def test(self)
        model.test()
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/IQA_model.py def test(self)
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
        # Process No.1201 Image and IQA_score: 0.5661910772323608
        # print(dataset_dir+'/{}.txt'.format(test_name), str(index), str(score))
        # /public/liuhongzhi/Method/IQA/SEU-IQA/results/LPIPS-Alex_TonPIPAL/test_PIPAL_Full_2024-07-31-20-11-37/test_PIPAL_Full.txt 
        # 5 0.671637773513794
        # time.sleep(5)
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
        f.write("and report single core CPU runtime. The method was trained on PIPAL dataset" + "\n")
        

if __name__ == '__main__':
    main()