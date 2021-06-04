import model
import data_create
import argparse
import os
import numpy as np
import tensorflow as tf

# gpu_id = 0
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow compare_super_resolution Example')
    parser.add_argument('--test_height', type = int, default = 720, help = "Test data HR size(height)")
    parser.add_argument('--test_width', type = int, default = 1280, help = "Test data HR size(width)")
    parser.add_argument('--test_dataset_num', type = int, default = 50, help = "Number of test datasets to generate")
    parser.add_argument('--test_cut_num', type = int, default = 1, help = "Number of test data to be generated from a single image")
    parser.add_argument('--test_path', type = str, default = "../../dataset/reds_val_sharp", help = "The path containing the test image")
    parser.add_argument('--LR_num', type = int, default = 5, help = "Number of LR frames")

    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)

    parser.add_argument('--mode', type=str, default='train_model', help='train_datacreate, test_datacreate, train_model, evaluate')

    args = parser.parse_args()

    if args.mode == 'test_datacreate': #Create evaluate datasets
        test_same_x, test_dif_x, test_y = data_create.datacreate().train_datacreate(args.test_path,#Path where training data is stored
                                            args.test_dataset_num,#Number of train datasets
                                            args.test_cut_num,#Number of data to be generated from a single image
                                            args.test_height,#Save size
                                            args.test_width)
        path = "test_data_list"
        np.savez(path, test_same_x, test_dif_x, test_y)

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