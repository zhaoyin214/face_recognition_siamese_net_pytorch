config = dict(train_set_root = "e:/src/jupyter/datasets/CASIA-WebFace",
              test_set_root = "e:/src/jupyter/datasets/ORL-Face",
              train_batch_size = 64,
              train_epochs = 100,
              is_rgb = True,
              dim_embedding = 256,
              )

if __name__ == "__main__":

    for key in config.keys():
        print(type(key))
        print("key: {}, value: {}".format(key, config[key]))