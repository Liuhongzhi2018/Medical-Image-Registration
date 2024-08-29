import os
import sys
import time
 
# cmd = 'CUDA_VISIBLE_DEVICES=0 nohup bash ../run.sh --stage 6'  #当GPU空闲时需要跑的脚本

# cmd = 'CUDA_VISIBLE_DEVICES=2 python tools/train.py configs_coco/cascade_rcnn/cascade-rcnn_x101_64x4d_fpn_20e_coco.py --work-dir ./work_dirs/cascade-rcnn_x101_64x4d_fpn_20e_coco_0712'

# (mmdet) liuhongzhi@user-SYS-7049GP-TRT:/mnt/lhz/Github/MedicalDetection/mmdetection$ 
# CUDA_VISIBLE_DEVICES=3 python tools/train.py configs_coco/cascade_rcnn/cascade-rcnn_x101_64x4d_fpn_20e_coco.py --work-dir ./work_dirs/cascade-rcnn_x101_64x4d_fpn_20e_coco_0907

# CUDA_VISIBLE_DEVICES=3 python tools/train.py configs_coco/cascade_rcnn/cascade-rcnn_r101_fpn_20e_coco.py --work-dir ./work_dirs/cascade-rcnn_r101_fpn_20e_coco_0907
cmd = 'CUDA_VISIBLE_DEVICES=2 python tools/train.py configs_coco/cascade_rcnn/cascade-rcnn_r101_fpn_20e_coco.py --work-dir ./work_dirs/cascade-rcnn_r101_fpn_20e_coco_0907'

# cmd = 'python helloworld.py'
 
# def gpu_info():
#     gpu_status = os.popen('nvidia-smi | grep %').read().split('|') #根据nvidia-smi命令的返回值按照'|'为分隔符建立一个列表
#     '''
#     结果如：
#     ['', ' N/A   64C    P0    68W /  70W ', '   9959MiB / 15079MiB ', '     79%      Default ', 
#     '\n', ' N/A   73C    P0   108W /  70W ', '  11055MiB / 15079MiB ', '     63%      Default ', 
#     '\n', ' N/A   60C    P0    55W /  70W ', '   3243MiB / 15079MiB ', '     63%      Default ', '\n']
#     '''
#     gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip()) 
#     #获取当前0号GPU功率值：提取标签为2的元素，按照'/'为分隔符后提取标签为0的元素值再按照'M'为分隔符提取标签为0的元素值，返回值为int形式 
#     gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
#     #获取0号GPU当前显存使用量
#     gpu_util = int(gpu_status[3].split('   ')[1].split('%')[0].strip())
#     #获取0号GPU显存核心利用率
#     return gpu_power, gpu_memory, gpu_util

# def all_gpu_info_2(p):
#     gpu_status = os.popen('nvidia-smi | grep %').read().split('|') #根据nvidia-smi命令的返回值按照'|'为分隔符建立一个列表
#     # print("\ngpu_status: ", gpu_status)
#     '''
#     结果如：
#     ['', ' N/A   64C    P0    68W /  70W ', '   9959MiB / 15079MiB ', '     79%      Default ', 
#     '\n', ' N/A   73C    P0   108W /  70W ', '  11055MiB / 15079MiB ', '     63%      Default ', 
#     '\n', ' N/A   60C    P0    55W /  70W ', '   3243MiB / 15079MiB ', '     63%      Default ', '\n']
    
#     ['', ' 71%   67C    P2   344W / 350W ', '  24244MiB / 24576MiB ', '    100%      Default ', 
#     '\n', ' 72%   68C    P2   345W / 350W ', '  24190MiB / 24576MiB ', '    100%      Default ', 
#     '\n', ' 62%   62C    P2   264W / 350W ', '  21822MiB / 24576MiB ', '    100%      Default ', 
#     '\n', ' 73%   69C    P2   340W / 350W ', '  23230MiB / 24576MiB ', '     99%      Default ', '\n']
#     '''
#     gpu_memory = int(gpu_status[p+1].split('/')[0].split('M')[0].strip()) 
#     #获取当前0号GPU功率值：提取标签为2的元素，按照'/'为分隔符后提取标签为0的元素值再按照'M'为分隔符提取标签为0的元素值，返回值为int形式 
#     gpu_power = int(gpu_status[p].split('   ')[-1].split('/')[0].split('W')[0].strip())
#     #获取0号GPU当前显存使用量
#     gpu_util = int(gpu_status[p+2].split('   ')[1].split('%')[0].strip())
#     #获取0号GPU显存核心利用率
#     return gpu_power, gpu_memory, gpu_util

# def all_gpu_info_3(p):
#     gpu_status = os.popen('nvidia-smi | grep %').read().split('|') #根据nvidia-smi命令的返回值按照'|'为分隔符建立一个列表
#     # print("\ngpu_status: ", gpu_status)
#     '''
#     结果如：
#     ['', ' N/A   64C    P0    68W /  70W ', '   9959MiB / 15079MiB ', '     79%      Default ', 
#     '\n', ' N/A   73C    P0   108W /  70W ', '  11055MiB / 15079MiB ', '     63%      Default ', 
#     '\n', ' N/A   60C    P0    55W /  70W ', '   3243MiB / 15079MiB ', '     63%      Default ', '\n']
    
#     ['', ' 71%   67C    P2   344W / 350W ', '  24244MiB / 24576MiB ', '    100%      Default ', 
#     '\n', ' 72%   68C    P2   345W / 350W ', '  24190MiB / 24576MiB ', '    100%      Default ', 
#     '\n', ' 62%   62C    P2   264W / 350W ', '  21822MiB / 24576MiB ', '    100%      Default ', 
#     '\n', ' 73%   69C    P2   340W / 350W ', '  23230MiB / 24576MiB ', '     99%      Default ', '\n']
#     '''
#     gpu_memory = float(gpu_status[p+1].split('/')[0].split('M')[0].strip()) 
#     #获取当前0号GPU功率值：提取标签为2的元素，按照'/'为分隔符后提取标签为0的元素值再按照'M'为分隔符提取标签为0的元素值，返回值为int形式 
#     gpu_power = float(gpu_status[p].split('   ')[-1].split('/')[0].split('W')[0].strip())
#     #获取0号GPU当前显存使用量
#     gpu_util = float(gpu_status[p+2].split('   ')[1].split('%')[0].strip())
#     #获取0号GPU显存核心利用率
#     return gpu_power, gpu_memory, gpu_util

def all_gpu_info_5(p):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|') #根据nvidia-smi命令的返回值按照'|'为分隔符建立一个列表
    # print("\ngpu_status: ", gpu_status)
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
    gpu_memory = float(gpu_status[p+1].split('/')[0].split('M')[0].strip()) 
    #获取当前0号GPU功率值：提取标签为2的元素，按照'/'为分隔符后提取标签为0的元素值再按照'M'为分隔符提取标签为0的元素值，返回值为int形式 
    gpu_power = float(gpu_status[p].split('   ')[-1].split('/')[0].split('W')[0].strip())
    #获取0号GPU当前显存使用量
    return gpu_power, gpu_memory
 
 
# def narrow_setup(secs=600):  #间隔十分钟检测一次
#     gpu_power, gpu_memory, gpu_util = gpu_info()
#     i = 0
#     while not(gpu_memory < 1000 and gpu_power < 20 and gpu_util < 20) :  # 当功率，使用量，利用率都小于特定值才去退出循环
        
#         gpu_power, gpu_memory, gpu_util = gpu_info()
#         i = i % 5
#         symbol = 'monitoring: ' + '> GPU' * i + ' ' * (10 - i - 1) + '|'
#         gpu_power_str = 'gpu power:%d W |' % gpu_power
#         gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
#         gpu_util_str = 'gpu util:%d %% |' % gpu_util
#         sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + gpu_util_str + ' ' + symbol)
#         #sys.stdout.write(obj+'\n')等价于print(obj)
#         sys.stdout.flush()    #刷新输出
#         time.sleep(secs)  #推迟调用线程的运行，通过参数指秒数，表示进程挂起的时间。
#         i += 1
#     print('\n' + cmd)
#     os.system(cmd) #执行脚本
    
    
# def run_process(secs=1):  #间隔十分钟检测一次
#     p = 1
#     gpu_power, gpu_memory, gpu_util = all_gpu_info_2(p)
#     i = 0
#     while not(gpu_memory < 17000) :  # 当功率，使用量，利用率都小于特定值才去退出循环
#         if p > 15:
#             p = 1
#         gpu_power, gpu_memory, gpu_util = all_gpu_info_2(p)
#         i = i % 5
#         gpu_index = 'GPU: ' + str(int(p//4))
#         symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
#         gpu_power_str = 'gpu power: %d W |' % gpu_power
#         gpu_memory_str = 'gpu memory: %d MiB |' % gpu_memory
#         gpu_util_str = 'gpu util: %d %% |' % gpu_util
#         sys.stdout.write('\r' + gpu_index + ' ' + gpu_memory_str + ' ' + gpu_power_str + ' ' + gpu_util_str + ' ' + symbol)
#         #sys.stdout.write(obj+'\n')等价于print(obj)
#         sys.stdout.flush()    #刷新输出
#         time.sleep(secs)  #推迟调用线程的运行，通过参数指秒数，表示进程挂起的时间。
#         current_p = str(int(p//4))
#         i += 1
#         p += 4

#     cmd = 'CUDA_VISIBLE_DEVICES=' + current_p + ' ' + 'training 6 3d_fullres 2'
#     print('\n' + cmd)
#     os.system(cmd) #执行脚本

# def run_process_cmd(secs=1):  #间隔十分钟检测一次
#     p = 1
#     gpu_power, gpu_memory, gpu_util = all_gpu_info_3(p)
#     i = 0
#     while not(gpu_memory < 7000) :  # 当功率，使用量，利用率都小于特定值才去退出循环
#         if p > 15:
#             p = 1
#         gpu_power, gpu_memory, gpu_util = all_gpu_info_3(p)
#         i = i % 5
#         gpu_index = 'GPU: ' + str(int(p//4))
#         symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
#         gpu_power_str = 'gpu power: %d W |' % gpu_power
#         gpu_memory_str = 'gpu memory: %d MiB |' % gpu_memory
#         gpu_util_str = 'gpu util: %d %% |' % gpu_util
#         # sys.stdout.write('\r' + gpu_index + ' ' + gpu_memory_str + ' ' + gpu_power_str + ' ' + gpu_util_str + ' ' + symbol)
#         sys.stdout.write('\r' + gpu_index + ' ' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
#         #sys.stdout.write(obj+'\n')等价于print(obj)
#         sys.stdout.flush()    #刷新输出
#         time.sleep(secs)  #推迟调用线程的运行，通过参数指秒数，表示进程挂起的时间。
#         current_p = str(int(p//4))
#         i += 1
#         p += 4
        
#         if gpu_memory < 7000 and current_p =='3':
#             break

#     cmd = 'CUDA_VISIBLE_DEVICES=' + current_p + ' ' + 'training 6 3d_fullres 2'
#     print('\n' + cmd)
#     os.system(cmd) #执行脚本


# def run_process_cmd2(secs=1):  #间隔十分钟检测一次
#     p = 1
#     gpu_power, gpu_memory = all_gpu_info_5(p)
#     i = 0
#     while not(gpu_memory < 17000) :  # 当功率，使用量，利用率都小于特定值才去退出循环
#         if p > 15:
#             p = 1
#         # gpu_power, gpu_memory, gpu_util = all_gpu_info_3(p)
#         gpu_power, gpu_memory = all_gpu_info_5(p)
#         i = i % 5
#         gpu_index = 'GPU: ' + str(int(p//4))
#         symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
#         gpu_power_str = 'gpu power: %d W |' % gpu_power
#         gpu_memory_str = 'gpu memory: %d MiB |' % gpu_memory
#         # gpu_util_str = 'gpu util: %d %% |' % gpu_util
#         sys.stdout.write('\r' + gpu_index + ' ' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
#         #sys.stdout.write(obj+'\n')等价于print(obj)
#         sys.stdout.flush()    #刷新输出
#         time.sleep(secs)  #推迟调用线程的运行，通过参数指秒数，表示进程挂起的时间。
#         current_p = str(int(p//4))
#         i += 1
#         p += 4

#     cmd = 'CUDA_VISIBLE_DEVICES=' + current_p + ' ' + 'training 6 3d_fullres 4'
#     print('\n' + cmd)
#     os.system(cmd) #执行脚本
    
    
def run_process_cmd_GPU2(secs=1):
    p = 1
    i = 0
    gpu_power, gpu_memory = all_gpu_info_5(p)
    while not(gpu_memory < 7000) :  # 当功率，使用量，利用率都小于特定值才去退出循环
        if p > 15:
            p = 1
        gpu_power, gpu_memory = all_gpu_info_5(p)
        i = i % 5
        gpu_index = 'GPU: ' + str(int(p//4))
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power: %d W |' % gpu_power
        gpu_memory_str = 'gpu memory: %d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_index + ' ' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()    #刷新输出
        time.sleep(secs)  #推迟调用线程的运行，通过参数指秒数，表示进程挂起的时间。
        current_p = str(int(p//4))
        i += 1
        p += 4
        
        if gpu_memory < 18000 and current_p =='2':
            break

    # cmd = 'CUDA_VISIBLE_DEVICES=' + current_p + ' ' + 'training 6 3d_fullres 2'
    print('\n' + cmd)
    os.system(cmd) #执行脚本
 
 
if __name__ == '__main__':
    # narrow_setup()
    # run_process()
    # run_process_cmd()
    # run_process_cmd2()
    run_process_cmd_GPU2()

