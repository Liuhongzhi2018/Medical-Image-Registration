#! /usr/bin/python3
import os
import time
import sys
import torch

class GPUGet:
    def __init__(self,
                 min_gpu_number,
                 time_interval):
        self.min_gpu_number = min_gpu_number
        self.time_interval = time_interval

    def get_gpu_info(self):
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')[1:]
        gpu_dict = dict()
        for i in range(len(gpu_status) // 4):
            index = i * 4
            gpu_state = str(gpu_status[index].split('   ')[2].strip())
            gpu_power = int(gpu_status[index].split('   ')[-1].split('/')[0].split('W')[0].strip())
            gpu_memory = int(gpu_status[index + 1].split('/')[0].split('M')[0].strip())
            gpu_dict[i] = (gpu_state, gpu_power, gpu_memory)
        return gpu_dict

    # def loop_monitor(self):
    #     available_gpus = []
    #     while True:
    #         gpu_dict = self.get_gpu_info()
    #         for i, (gpu_state, gpu_power, gpu_memory) in gpu_dict.items():
    #             # if gpu_state == "P8" and gpu_power <= 40 and gpu_memory <= 1000:  # 设置GPU选用条件，当前适配的是Nvidia-RTX3090
    #             if gpu_memory <= 1000:  # 设置GPU选用条件，当前适配的是Nvidia-RTX3090                    
    #                 gpu_str = f"GPU/id: {i}, GPU/state: {gpu_state}, GPU/memory: {gpu_memory}MiB, GPU/power: {gpu_power}W\n "
    #                 # print(gpu_str)
    #                 sys.stdout.write(gpu_str)
    #                 sys.stdout.flush()
    #                 available_gpus.append(i)
    #         if len(available_gpus) >= self.min_gpu_number:
    #             return available_gpus
    #         else:
    #             available_gpus = []
    #             time.sleep(self.time_interval)

    # def run(self, cmd_parameter, cmd_command):
    #     available_gpus = self.loop_monitor()
    #     gpu_list_str = ",".join(map(str, available_gpus))
    #     # 构建终端命令
    #     cmd_parameter = fr"""{cmd_parameter}
    #                       NUM_GPUS={len(available_gpus)} ; \ """  # 一定要有 `; \ `
    #     cmd_command = fr"""CUDA_VISIBLE_DEVICES={gpu_list_str} \ 
    #                      {cmd_command}"""
    #     command = fr"""{cmd_parameter} {cmd_command}"""
    #     # print(command)
    #     # os.system(command)
    #     command_run = fr"""CUDA_VISIBLE_DEVICES={gpu_list_str} \ 
    #                      {cmd_command}"""
    #     print(command_run)
        os.system(command_run)
        
    def loop_monitor_realtime(self):
        available_gpus = []
        while True:
            gpu_dict = self.get_gpu_info()
            for i, (gpu_state, gpu_power, gpu_memory) in gpu_dict.items():
                gpu_now = f"GPU/id: {i}, GPU/state: {gpu_state}, GPU/memory: {gpu_memory}MiB, GPU/power: {gpu_power}W\n "
                print(gpu_now)
                # if gpu_state == "P8" and gpu_power <= 40 and gpu_memory <= 1000:  # 设置GPU选用条件，当前适配的是Nvidia-RTX3090
                if gpu_memory <= 10000:  # 设置GPU选用条件，当前适配的是Nvidia-RTX3090                    
                    gpu_str = f"GPU/id: {i}, GPU/state: {gpu_state}, GPU/memory: {gpu_memory}MiB, GPU/power: {gpu_power}W\n "
                    # print(gpu_str)
                    sys.stdout.write(gpu_str)
                    sys.stdout.flush()
                    available_gpus.append(i)
            if len(available_gpus) >= self.min_gpu_number:
                return available_gpus
            else:
                available_gpus = []
                time.sleep(self.time_interval)
        
    def run_code(self):
        available_gpus = self.loop_monitor_realtime()
        gpu_list_str = ",".join(map(str, available_gpus))
        # gpu_list_str = [2]
        print("gpu_list_str: ", gpu_list_str)
        if int(gpu_list_str[0])>=0:
            while 1:
                # a = torch.zeros((1500, 1000, 600), dtype=torch.float64, requires_grad=True).cuda(int(gpu_list_str))+200
                # b = torch.randn((1500, 600, 200), dtype=torch.float64, requires_grad=True).cuda(int(gpu_list_str))+200
                
                # # 22GB
                # a = torch.zeros((1500, 1200, 800), dtype=torch.float64).cuda(int(gpu_list_str[0]))+200
                # b = torch.randn((1500, 800, 500), dtype=torch.float64).cuda(int(gpu_list_str[0]))+200
                
                # # 10GB
                # a = torch.zeros((1000, 1200, 800), dtype=torch.float64).cuda(int(gpu_list_str[0]))+200
                # b = torch.randn((1000, 800, 500), dtype=torch.float64).cuda(int(gpu_list_str[0]))+200
                
                # 10GB
                a = torch.zeros((300, 1200, 800), dtype=torch.float64).cuda(int(gpu_list_str[0]))+200
                b = torch.randn((300, 800, 500), dtype=torch.float64).cuda(int(gpu_list_str[0]))+200
                
                # z = torch.matmul(a, b)
                # print(f"z shape {z.shape}")
                time.sleep(864000)
                del a, b, z
                # torch.cuda.empty_cache()
                # del a, b
                torch.cuda.empty_cache()
                
                # a = torch.zeros((1200, 1000, 600), dtype=torch.float64).cuda(int(gpu_list_str))+200
                # b = torch.randn((1200, 600, 500), dtype=torch.float64).cuda(int(gpu_list_str))+200


if __name__ == '__main__':
    # conda activate py38pt11
    min_gpu_number = 1  # 最小GPU数量，多于这个数值才会开始执行训练任务。
    time_interval = 1  # 监控GPU状态的频率，单位秒。
    gpu_get = GPUGet(min_gpu_number, time_interval)

    cmd_parameter = r""""""  # 命令会使用到的参数，使用 `;` 连接。
    cmd_command = r"""tools/dist_train.sh ${NUM_GPUS} \ """
    # gpu_get.run(cmd_parameter, cmd_command)
    gpu_get.run_code()