import os
from PIL import Image

if __name__ == '__main__':
    group_dic = {
        '0' : 'Accessorie',
        '1' : 'Armor',
        '2' : 'Food',
        '3' : 'Material',
        '4' : 'Shield',
        '5' : 'Special_Item',
        '6' : 'Supply',
        '7' : 'Weapon'
    }

    fold_list = []

    #找到所有数据库里存放Icon的文件夹的名字
    for root, dirs, files in os.walk('./Dataset/Group_Icon'):
        if(len(dirs) >= 8):
            fold_list = dirs #当前路径下所有子目录
    print(fold_list)
    
    #找到所有文件名，二维数组保存
    files_list = []
    for i in range(len(fold_list)):
        files_list.append(os.listdir('./Dataset/Group_Icon/'+fold_list[i]))

    print(files_list)
    
    #根据上述两个列表还原出所有Icon的文件路径
    file_paths = []
    for i in range(len(fold_list)):
        for j in range(len(files_list[i])):
            file_paths.append('./Dataset/' + fold_list[i] + '/' + files_list[i][j])

    #读取并写入文件
    for i in range(len(fold_list)):
        for j in range(len(files_list[i])):
            #最后的命名将会是0_123.png,0代表第一类，123代表第123张图
            #file_name = str(fold_list[i].split('.')[0]) + '_' + files_list[i][j]
            file_name = str(i) + '_' + files_list[i][j]
            with Image.open('./Dataset/Group_Icon/' + fold_list[i] + '/' + files_list[i][j]) as image:
                image = image.resize((128, 128))
                image.save('./Group_Icon/'+file_name)
            print(file_name)

    