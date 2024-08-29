import os
import sys
import time
 
# cmd = 'CUDA_VISIBLE_DEVICES=0 nohup bash ../run.sh --stage 6'  #当GPU空闲时需要跑的脚本

# cmd = 'CUDA_VISIBLE_DEVICES=2 python tools/train.py configs_coco/cascade_rcnn/cascade-rcnn_x101_64x4d_fpn_20e_coco.py --work-dir ./work_dirs/cascade-rcnn_x101_64x4d_fpn_20e_coco_0712'

cmd = 'CUDA_VISIBLE_DEVICES=2 python tools/train.py configs_voc/faster_rcnn/faster-rcnn_r101_fpn_1x_voc.py --work-dir ./work_dirs/faster-rcnn_r101_fpn_1x_voc_0703'
 
def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|') #根据nvidia-smi命令的返回值按照'|'为分隔符建立一个列表
    '''
    结果如：
    ['', ' N/A   64C    P0    68W /  70W ', '   9959MiB / 15079MiB ', '     79%      Default ', 
    '\n', ' N/A   73C    P0   108W /  70W ', '  11055MiB / 15079MiB ', '     63%      Default ', 
    '\n', ' N/A   60C    P0    55W /  70W ', '   3243MiB / 15079MiB ', '     63%      Default ', '\n']
    '''
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip()) 
    #获取当前0号GPU功率值：提取标签为2的元素，按照'/'为分隔符后提取标签为0的元素值再按照'M'为分隔符提取标签为0的元素值，返回值为int形式 
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    #获取0号GPU当前显存使用量
    gpu_util = int(gpu_status[3].split('   ')[1].split('%')[0].strip())
    #获取0号GPU显存核心利用率
    return gpu_power, gpu_memory, gpu_util

def all_gpu_info_2(p):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|') #根据nvidia-smi命令的返回值按照'|'为分隔符建立一个列表
    print("gpu_status: ", gpu_status)
    '''
    结果如：
    ['', ' N/A   64C    P0    68W /  70W ', '   9959MiB / 15079MiB ', '     79%      Default ', 
    '\n', ' N/A   73C    P0   108W /  70W ', '  11055MiB / 15079MiB ', '     63%      Default ', 
    '\n', ' N/A   60C    P0    55W /  70W ', '   3243MiB / 15079MiB ', '     63%      Default ', '\n']
    
    ['', ' 71%   67C    P2   344W / 350W ', '  24244MiB / 24576MiB ', '    100%      Default ', 
    '\n', ' 72%   68C    P2   345W / 350W ', '  24190MiB / 24576MiB ', '    100%      Default ', 
    '\n', ' 62%   62C    P2   264W / 350W ', '  21822MiB / 24576MiB ', '    100%      Default ', 
    '\n', ' 73%   69C    P2   340W / 350W ', '  23230MiB / 24576MiB ', '     99%      Default ', '\n']
    '''
    gpu_memory = int(gpu_status[p+1].split('/')[0].split('M')[0].strip()) 
    #获取当前0号GPU功率值：提取标签为2的元素，按照'/'为分隔符后提取标签为0的元素值再按照'M'为分隔符提取标签为0的元素值，返回值为int形式 
    gpu_power = int(gpu_status[p].split('   ')[-1].split('/')[0].split('W')[0].strip())
    #获取0号GPU当前显存使用量
    gpu_util = int(gpu_status[p+2].split('   ')[1].split('%')[0].strip())
    #获取0号GPU显存核心利用率
    return gpu_power, gpu_memory, gpu_util
 
 
def narrow_setup(secs=600):  #间隔十分钟检测一次
    gpu_power, gpu_memory, gpu_util = gpu_info()
    i = 0
    while not(gpu_memory < 1000 and gpu_power < 20 and gpu_util < 20) :  # 当功率，使用量，利用率都小于特定值才去退出循环
        
        gpu_power, gpu_memory, gpu_util = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        gpu_util_str = 'gpu util:%d %% |' % gpu_util
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + gpu_util_str + ' ' + symbol)
        #sys.stdout.write(obj+'\n')等价于print(obj)
        sys.stdout.flush()    #刷新输出
        time.sleep(secs)  #推迟调用线程的运行，通过参数指秒数，表示进程挂起的时间。
        i += 1
    print('\n' + cmd)
    os.system(cmd) #执行脚本
    
    
def run_process(secs=1):  #间隔十分钟检测一次
    p = 1
    gpu_power, gpu_memory, gpu_util = all_gpu_info_2(p)
    i = 0
    while not(gpu_memory < 22000) :  # 当功率，使用量，利用率都小于特定值才去退出循环
        if p > 15:
            p = 1
        gpu_power, gpu_memory, gpu_util = all_gpu_info_2(p)
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        gpu_util_str = 'gpu util:%d %% |' % gpu_util
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + gpu_util_str + ' ' + symbol)
        #sys.stdout.write(obj+'\n')等价于print(obj)
        sys.stdout.flush()    #刷新输出
        time.sleep(secs)  #推迟调用线程的运行，通过参数指秒数，表示进程挂起的时间。
        i += 1
        p += 4
    print('\n' + cmd)
    os.system(cmd) #执行脚本
 
 
if __name__ == '__main__':
    # narrow_setup()
    run_process()