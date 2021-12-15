Step1:将json(https://drive.google.com/file/d/15SnEOy_g0Hk8FvUG9vZtVnS6y_q0TnNA/view?usp=sharing)文件放入caption data文件夹内。
将图片(https://drive.google.com/file/d/1GULOcQhXiaXQ9nD3kvS_AobJ8l84zP97/view?usp=sharing)解压到media/ssd/caption data文件夹内，然后运行create_input_files.py创建模型训练所需的输入。
将会在media/ssd/caption data下面生成一系列文件，请记住WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json的路径。

Step2:运行train.py。记得调整epoch，我手动设置为了1。训练完毕后会自动保存。

Step3:命令行使用 python caption.py -i 图片路径 -m 模型路径 -w WORDMAP路径 即可获取字幕输
(如果不想训练可以直接使用我的模型和我的WORDMAP,直接一起放到根目录底下就行)
https://drive.google.com/file/d/14qxSzbmu9jBJxVe3WqBwB4oaCZLvco4t/view?usp=sharing
https://drive.google.com/file/d/1ZmUhyI6s479qnOJMmfWBMauUDTKn667B/view?usp=sharing