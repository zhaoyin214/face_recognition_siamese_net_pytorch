# config = dict(training_set_dir = "../../../datasets/CASIA-WebFace",
#               test_set_dir = "./代码数据分享/data/faces/testing",
#               training_batch_size = 256,
#               training_epochs = 100,
#               is_rgb = True,
#               dim_embedding = 256,
#               )


config = dict(training_set_dir = "./代码数据分享/data/faces/training",
              test_set_dir = "./代码数据分享/data/faces/testing",
              training_batch_size = 64,
              training_epochs = 100,
              is_rgb = False,
              dim_embedding = 256,
              )

if __name__ == "__main__":

    for key in config.keys():
        print(type(key))
        print("key: {}, value: {}".format(key, config[key]))