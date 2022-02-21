import os
from os import getcwd  #输入os这个包，os这个包一般指系统路径，类比ios

#---------------------------------------------------#
#   训练自己的数据集的时候一定要注意修改classes
#   修改成自己数据集所区分的种类
#   
#   种类顺序需要和训练时用到的model_data下的txt一样
#---------------------------------------------------#
classes = ["disgust", "fear", "happiness", "others", "repression", "sadness", "surprise"]  #分类
sets    = ["train", "test"]  #就是指我们训练和测试的两个集

if __name__ == "__main__":  #输入主函数
    wd = getcwd()     #返回当前路径,就是该代码所处的文件夹路径
    print(wd)
    for se in sets:  #循环语句，第一次循环train，第二次循环test。第一次循环时se是什么train
        list_file = open('cls_' + se + '.txt', 'w')  #cls_train.txt txt为后缀名，可以类比一张图像，1.png 'w'代表write ‘r’代表read

        datasets_path = "datasets/" + se  #datasets/train
        types_name = os.listdir(datasets_path)  #os.listdir就是把当前的文件夹里面的所有文件夹名称打印出来
        print(types_name)
        for type_name in types_name: #再一次循环 循环7次 第一次循环disgust type_name：第一次为disgust
            if type_name not in classes:  # 判断语句
                continue
            cls_id = classes.index(type_name)  # classes.index按顺序输出0123456..... 就是把gisgust这一类,标签设定为0
            print(cls_id)
            photos_path = os.path.join(datasets_path, type_name) #os.path.join 就是将datasets_path, type_name路径结合在一起   datasets/train/disgust
            photos_name = os.listdir(photos_path)  #os.listdir就是把当前的文件夹里面的所有文件夹名称打印出来   photos_name照片名字
            # print(photos_name)
            for photo_name in photos_name:#循环，对照片一一提取 photo_name代表第一张照片
                _, postfix = os.path.splitext(photo_name)  #os.path.splitext 假如我们有一个照片为7560.jpg ，那么执行这行代码后的输出为两个，前面为7560，后面为.jpg
                if postfix not in ['.jpg', '.png', '.jpeg']: #一个判断语句
                    continue
                list_file.write(str(cls_id) + ";" + '%s/%s'%(wd, os.path.join(photos_path, photo_name)))  # 0;E:\Hom_workspace\casme2\classification-keras-release\datasets\train/disgust/照片名
                list_file.write('\n')
        list_file.close()

