import nvidia_smi as nvs
nvs.nvmlInit()

def get_logger(log_path):
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(log_path, 'a+') as f_log:
                f_log.write(s + '\n')
    return logging

def get_gpu_usage(gpu_id):
    gpu = nvs.nvmlDeviceGetHandleByIndex(gpu_id)
    usage = nvs.nvmlDeviceGetMemoryInfo(gpu)
    return usage.used / usage.total
