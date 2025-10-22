#!/bin/bash

# 读取参数
input_config=$1
gen_train_data_root=$2
exp_name=$3

# 输出配置文件名
output_config="${input_config%.py}_$exp_name.py"

# 创建一个新的配置文件
cp "$input_config" "$output_config"

# 使用sed命令修改配置文件
sed -i "s|gen_train_data_root = .*$|gen_train_data_root = '$gen_train_data_root'|" "$output_config"

echo "Modified config file has been saved as $output_config."

# 脚本结束