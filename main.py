import model
import data_create
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow DRCN Example')

    parser.add_argument('--train_height', type = int, default = 50, help = "Train data HR size(height)")
    parser.add_argument('--train_width', type = int, default = 50, help = "Train data HR size(width)")
    parser.add_argument('--test_height', type = int, default = 360, help = "Test data HR size(height)")
    parser.add_argument('--test_width', type = int, default = 640, help = "Test data HR size(width)")
    parser.add_argument('--train_dataset_num', type = int, default = 30000, help = "Number of train datasets to generate")
    parser.add_argument('--test_dataset_num', type = int, default = 5, help = "Number of test datasets to generate")
    parser.add_argument('--train_cut_num', type = int, default = 10, help = "Number of train data to be generated from a single image")
    parser.add_argument('--test_cut_num', type = int, default = 1, help = "Number of test data to be generated from a single image")
    parser.add_argument('--train_path', type = str, default = "../../dataset/reds_train_sharp", help = "The path containing the train image")
    parser.add_argument('--test_path', type = str, default = "../../dataset/reds_val_sharp", help = "The path containing the test image")
    parser.add_argument('--LR_num', type = int, default = 5, help = "Number of LR frames")
    parser.add_argument('--learning_rate', type = float, default = 1e-4, help = "Learning_rate")
    parser.add_argument('--BATCH_SIZE', type = int, default = 32, help = "Training batch size")
    parser.add_argument('--EPOCHS', type = int, default = 1000, help = "Number of epochs to train for")
   
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)

    parser.add_argument('--mode', type=str, default='train_model', help='train_datacreate, test_datacreate, train_model, evaluate')

    args = parser.parse_args()

    if args.mode == 'train_datacreate': #Create train same and different size datasets
        train_same_x, train_dif_x, train_y = data_create.datacreate().train_datacreate(args.train_path,      #Path where training data is stored
                                            args.train_dataset_num,                                          #Number of train datasets
                                            args.train_cut_num,                                              #Number of data to be generated from a single image
                                            args.train_height,                                               #Save size
                                            args.train_width)   
        path = "train_data_list"
        np.savez(path, train_same_x, train_dif_x, train_y)

    elif args.mode == 'test_datacreate': #Create evaluate datasets
        test_x, test_y = data_create.datacreate().datacreate_different_size(args.test_path,
                                            args.test_dataset_num,
                                            args.test_cut_num,
                                            args.test_height,
                                            args.test_width)
        path = "test_data_differnt_size"
        np.savez(path, test_x, test_y)

        test_x, test_y = data_create.datacreate().datacreate_same_size(args.test_path,
                                            args.test_dataset_num,
                                            args.test_cut_num,
                                            args.test_height,
                                            args.test_width)
        path = "test_data_same_size"
        np.savez(path, test_x, test_y)

    elif args.mode == "train_srcnn": #train
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"]  #same size datasets
        train_y = npz["arr_2"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Single_SR().SRCNN()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit(train_x[args.LR_num // 2], train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)

        os.makedirs("model", exist_ok = True)
        train_model.save("model/SRCNN_model.h5")

    elif args.mode == "train_fsrcnn": #train
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_1"] #different size datasets
        train_y = npz["arr_2"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Single_SR().FSRCNN()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit(train_x[args.LR_num // 2], train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)

        os.makedirs("model", exist_ok = True)
        train_model.save("model/FSRCNN_model.h5")


    elif args.mode == "train_espcn": #train
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_1"] #different size dataset
        train_y = npz["arr_2"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Single_SR().ESPCN()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit(train_x[args.LR_num // 2], train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)
        os.makedirs("model", exist_ok = True)
        train_model.save("model/ESPCN_model.h5")

    elif args.mode == "train_vdsr": #train
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"] #same size datasets
        train_y = npz["arr_2"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Single_SR().VDSR()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit(train_x[args.LR_num // 2], train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)
        os.makedirs("model", exist_ok = True)
        train_model.save("model/VDSR_model.h5")

    elif args.mode == "train_drcn": #train

        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"] #same size datasets
        train_y = npz["arr_2"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Single_SR().DRCN()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit(train_x[args.LR_num // 2], train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)
        os.makedirs("model", exist_ok = True)
        train_model.save("model/DRCN_model.h5")

    elif args.mode == "train_red_net": #train
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"] #same size datasets
        train_y = npz["arr_2"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Single_SR().RED_Net()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit(train_x[args.LR_num // 2], train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)
        os.makedirs("model", exist_ok = True)
        train_model.save("model/RED_Net_model.h5")

    elif args.mode == "train_drrn": #train
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"] #same size datasets
        train_y = npz["arr_2"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Single_SR().RED_Net()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit(train_x[args.LR_num // 2], train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)
        os.makedirs("model", exist_ok = True)
        train_model.save("model/drrn_model.h5")

    elif args.mode == "train_vsrnet": #train
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"] #same size datasets
        train_y = npz["arr_2"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Video_SR().VSRnet()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit({"input_0" : train_x[args.LR_num // 2 -1], "input_1" : train_x[args.LR_num // 2], "input_2" : train_x[args.LR_num // 2 + 1]}, train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)
        os.makedirs("model", exist_ok = True)
        train_model.save("model/VSRnet_model.h5")

    elif args.mode == "train_rvsr": #train
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_1"] #different size datasets
        train_y = npz["arr_2"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Video_SR().RVSR()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit({"input_0" : train_x[args.LR_num // 2 -2], "input_1" : train_x[args.LR_num // 2 -1], "input_2" : train_x[args.LR_num // 2], "input_3" : train_x[args.LR_num // 2 +1], "input_4" : train_x[args.LR_num // 2 + 2]}, train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)
        os.makedirs("model", exist_ok = True)
        train_model.save("model/RVSR_model.h5")

    elif args.mode == "train_vespcn": #学習
        npz = np.load("train_data_differnt_size.npz")
        train_x = npz["arr_0"]
        train_y = npz["arr_1"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.Video_SR().VESPCN()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error", optimizer = optimizers, metrics = [psnr])

        train_model.fit({"input_0" : train_x[args.LR_num // 2 -1], "input_1" : train_x[args.LR_num // 2], "input_2" : train_x[args.LR_num // 2 + 1]}, train_y, epochs = args.EPOCHS, verbose = 2, batch_size = args.BATCH_SIZE)
        os.makedirs("model", exist_ok = True)
        train_model.save("model/VESPCN_model.h5")
        
    # elif args.mode == "srcnn_evaluate": #評価
    #     result_path = "result/SRCNN_result"
    #     os.makedirs(result_path, exist_ok = True)

    #     npz = np.load("test_data_same_size.npz", allow_pickle = True)

    #     test_x = npz["arr_0"]
    #     test_y = npz["arr_1"]

    #     test_x = tf.convert_to_tensor(test_x, np.float32)
    #     test_y = tf.convert_to_tensor(test_y, np.float32)

    #     test_x /= 255
    #     test_y /= 255
            
    #     path = "model/SRCNN_model.h5"

    #     if os.path.exists(path):
    #         model = tf.keras.models.load_model(path, custom_objects={'psnr':psnr})
    #         pred = model.predict(test_x, batch_size = 1)

    #         for p in range(len(test_y)):
    #             pred[p][pred[p] > 1] = 1
    #             pred[p][pred[p] < 0] = 0
    #             ps_pred = psnr(tf.reshape(test_y[p], [args.test_height, args.test_width, 1]), pred[p])
    #             ps_bicubic = psnr(tf.reshape(test_y[p], [args.test_height, args.test_width, 1]), tf.reshape(test_x[args.LR_num // 2][p], [args.test_height, args.test_width, 1]))

    #             low_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_x[args.LR_num // 2][p] * 255, [args.test_height, args.test_width]))
    #             cv2.imwrite(result_path + "/" + str(p) + "_low" + ".jpg", low_img)     #LR
    #             high_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_y[p] * 255, [args.test_height, args.test_width]))
    #             cv2.imwrite(result_path + "/" + str(p) + "_high" + ".jpg", high_img)   #HR
    #             pred_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(pred[p] * 255, [args.test_height, args.test_width]))
    #             cv2.imwrite(result_path + "/" + str(p) + "_pred" + ".jpg", pred_img)   #pred

    #             print("num:{}".format(p))
    #             print("psnr_pred:{}".format(ps_pred))
    #             print("psnr_bicubic:{}".format(ps_bicubic))