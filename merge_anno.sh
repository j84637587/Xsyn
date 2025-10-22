#!/bin/bash

# 读取参数
file1=$1
file2=$2
file3=$3
file4=$4
# file5=$5
# file6=$6
# file7=$7
# file8=$8
output_path=$5

# 输出文件名
output_file="$output_path/annotation.txt"


file1="$output_path/$file1"
file2="$output_path/$file2"
file3="$output_path/$file3"
file4="$output_path/$file4"
# file5="$output_path/$file5"
# file6="$output_path/$file6"
# file7="$output_path/$file7"
# file8="$output_path/$file8"


# 检查文件是否存在
# for file in "$file1" "$file2" "$file3" "$file4"; do
#     if [ ! -f "$file" ]; then
#         echo "Error: File '$file' not found."
#         exit 1
#     fi
# done


# 合并到新文件
cat "$file1" "$file2" "$file3" "$file4"  > "$output_file"


echo "Merged file has been created: $output_file."

# 脚本结束