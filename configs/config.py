config = dict(train_set_root = "e:/src/jupyter/datasets/CASIA-WebFace",
              train_batch_size = 64,
              train_epochs = 100,
              val_set_root = "e:/src/jupyter/datasets/ORL-Face",
              val_batch_size = 400,
              is_rgb = False,
              dim_embedding = 256,
              early_stopping_patience = 15,
              reduce_lr_on_plateau = dict(patience= 5, factor=0.2),
              )

# # test
# config = dict(train_set_root = "e:/src/jupyter/datasets/ORL-Face",
#               train_batch_size = 64,
#               train_epochs = 100,
#               val_set_root = "e:/src/jupyter/datasets/ORL-Face",
#               val_batch_size = 400,
#               is_rgb = False,
#               dim_embedding = 256,
#               early_stopping_patience = 15,
#               reduce_lr_on_plateau = dict(patience= 5, factor=0.2),
#               )

if __name__ == "__main__":

    for key in config.keys():
        print(type(key))
        print("key: {}, value: {}".format(key, config[key]))