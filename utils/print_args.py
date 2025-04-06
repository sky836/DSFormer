from utils.tools import print_log


def print_args(args, log):
    # 将 args 转换为字典
    args_dict = vars(args)
    
    # 计算最大参数名称的长度，用于对齐
    max_key_length = max(len(key) for key in args_dict.keys())
    
    # 美化打印输出
    print_log(log, "Arguments:")
    print_log(log, "-" * (max_key_length + 20))  # 输出分隔线
    for key, value in args_dict.items():
        print_log(log, f"{key:<{max_key_length}} : {value}")
    print_log(log, "-" * (max_key_length + 20))  # 输出分隔线