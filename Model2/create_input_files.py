from utils import create_input_files

if __name__ == '__main__':
    #调用create_input_files来创建相关数据，顺便检查一下是否已经可以开始训练
    #这个程序能跑就证明可以开始训练了
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='./caption data/dataset_flickr8k.json',
                       image_folder='./media/ssd/caption data/Images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./media/ssd/caption data/',
                       max_len=50)
