#!bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate condavenv 
echo 'conda env activated'
conda install python==3.7.6
conda install tensorflow
cd ./tf-pose-estimation
pip install -r requirements.txt
conda install swig
cd ./tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
pip install opencv-python
pip install git+https://github.com/adrianc-a/tf-slim.git@remove_contrib
cd ../..
cd models/graph/cmu
bash download.sh
cd ../../..
python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
#python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
