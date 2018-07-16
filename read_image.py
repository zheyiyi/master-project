import os
images_class = {}
path_all_data= './train'
path_all_validation = './val'
path_all_test = './test'


for class_dir in os.listdir(path_all_data):
    if class_dir != '.DS_Store':
        train_dir = os.path.join(path_all_data, class_dir)
        validation_dir = os.path.join(path_all_validation, class_dir)
        test_dir = os.path.join(path_all_test, class_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)

        total_images = os.listdir(train_dir)
        validation_images = total_images[:40]
        test_images = total_images[40:80]

        for image in test_images:
            if train_dir != './train/.DS_Store':
                image_dir = os.path.join(train_dir,image)
                image_test_dir = os.path.join(test_dir,image)
                with open(image_dir, 'rb') as fh:
                    with open(image_test_dir,'wb') as fd:
                        fd.write(fh.read())
                        print('write finished %s in test file' %image)
                os.remove(image_dir)

        for image in validation_images:
            if train_dir != './train/.DS_Store':
                image_dir = os.path.join(train_dir,image)
                image_validation_dir = os.path.join(validation_dir,image)
                with open(image_dir, 'rb') as fh:
                    with open(image_validation_dir,'wb') as fd:
                        fd.write(fh.read())
                        print('write %s in validation file' %image)
                os.remove(image_dir)

print('compelete!')







