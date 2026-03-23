#!/bin/bash

# 运行第一个 Python 脚本
echo "Running extract_images.py..."
python data_extraction/extract_images.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to run extract_images.py"
    exit 1
fi

# 运行第二个 Python 脚本
echo "Running extract_features.py..."
python data_extraction/extract_features.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to run extract_features.py"
    exit 1
fi

# 完成
echo "Pipeline completed successfully!"