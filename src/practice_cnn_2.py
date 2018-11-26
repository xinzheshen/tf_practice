'''
 卷积神经网络
 数据集： http://vision.stanford.edu/aditya86/ImageNetDogs/
'''

import tensorflow as tf
import glob
import os
from itertools import groupby
from collections import defaultdict


training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

source = '../Images/n02*/*.jpg'
image_filenames = glob.glob(source)
print('image_filenames0', image_filenames[0].split(os.path.sep))

image_filenames_with_breed = map(lambda filename: (filename.split(os.path.sep)[1], filename),
                                 image_filenames)

# 依据品种（image_filenames_with_breed元组中的第0各分量）对图像分组
for dog_breed, breed_images in groupby(image_filenames_with_breed, lambda x: x[0]):
    # 枚举每个品种的图像，并将大概20%的图像划入测试集
    for i, breed_image in enumerate(breed_images):
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])

    # 检查每个品种的测试图像是否至少有全部图像的18%
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])

    assert round(breed_testing_count / (breed_testing_count + breed_training_count), 2) > 0.18, \
        'Not enough testing images'


def write_record_file(dataset, record_location):
    '''
    用dataset中的图像填充一个TFRecord文件，并将其类包含进来

    :param dataset: dict(list)

    :param record_location: str
            存储TFRrcord输出的路径
    :return:
    '''
    writer = None
    sess = tf.Session()

    # 枚举dataset。因为当前索引用于对文件进行划分，每隔100幅图像，训练样本的信息就被写入到一个新的TFRecord文件中，以加快写操作的进程
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)

            current_index += 1
            image_file = tf.read_file(image_filename)

            # 在ImageNet的狗的图像中，有少量无法被tf识别为JPEG的图像
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print('file_name', image_filename)
                continue
            # 转换为灰度图可以减少处理的计算量和内存占用，但这不是必需的
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, [250, 151])

            # 这里之所以使用tf.cast，是因为虽然尺寸更改后的图像的数据类型是浮点型，但RGB值尚未转换到[0,1)区间
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

            # 将标签按字符串存储较高效，推荐的做法是将其转换为整数索引或读热编码的秩1张量
            image_label = breed.encode('utf-8')
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))

            writer.write(example.SerializeToString())

    writer.close()
    sess.close()


write_record_file(training_dataset, '../output/training-images/training-image')
write_record_file(testing_dataset, '../output/testing-images/testing-image')

print('end')

