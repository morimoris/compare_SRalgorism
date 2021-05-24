import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Input, Add, Multiply, Concatenate, Average, Lambda
import MyPReLU

#Single image super resolution
class Single_SR():
    def __init__(self):
        self.upsampling_scale = 2
        self.input_channels = 1
        """
        upsampling_scale : magnification
        input_channels : channels of input_img.(gray → 1, RGB → 3)
        """
    #SRCNN
    def SRCNN(self):
        #input single image
        input_shape = Input((None, None, self.input_channels))

        #convolution
        conv2d_0 = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_shape)
        conv2d_1 = Conv2D(filters = 32, kernel_size = (1, 1), padding = "same", activation = "relu")(conv2d_0)
        conv2d_2 = Conv2D(filters = self.input_channels, kernel_size = (5, 5), padding = "same")(conv2d_1)

        model = Model(inputs = input_shape, outputs = [conv2d_2])
        model.summary()

        return model

    def FSRCNN(self, d = 56, s = 16, m = 4): 
        """
        d : The LR feature dimension
        s : The number of shrinking filters
        m : The mapping depth
        mag : Magnification
        """
        input_shape = Input((None, None, self.input_channels))

        #Feature extraction
        conv2d_0 = Conv2D(filters = d, kernel_size = (5, 5), padding = "same", activation = MyPReLU.MyPReLU())(input_shape)
        
        #Shrinking
        conv2d_1 = Conv2D(filters = s, kernel_size = (1, 1), padding = "same", activation = MyPReLU.MyPReLU())(conv2d_0)
        
        #Mapping 
        conv2d_2 = conv2d_1
        for i in range(m):
            conv2d_2 = Conv2D(filters = s, kernel_size = (3, 3), padding = "same", activation = MyPReLU.MyPReLU())(conv2d_2)

        #Expanding
        conv2d_3 = Conv2D(filters = d, kernel_size = (1, 1), padding = "same", activation = MyPReLU.MyPReLU())(conv2d_2)

        #Deconvolution
        conv2d_4 = Conv2DTranspose(filters = self.input_channels, kernel_size = (9, 9), strides = (self.upsampling_scale, self.upsampling_scale), padding = "same")(conv2d_3)

        model = Model(inputs = input_shape, outputs = conv2d_4)
        model.summary()

        return model
        
    #ESPCN
    def ESPCN(self, upsampling_scale = 2):
        #input single image
        input_shape = Input((None, None, self.input_channels))

        #convolution
        conv2d_0 = Conv2D(filters = 64, kernel_size = (5, 5), padding = "same", activation = "relu")(input_shape)
        conv2d_1 = Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu")(conv2d_0)
        conv2d_2 = Conv2D(filters = upsampling_scale ** 2, kernel_size = (3, 3), padding = "same")(conv2d_1)

        #pixel_shuffle
        pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, upsampling_scale))(conv2d_2)

        model = Model(inputs = input_shape, outputs = pixel_shuffle)
        model.summary()

        return model

    #VDSR
    def VDSR(self, depth = 20): 
        """
        depth : number of residual blocks.
        """
        #input single image
        input_shape = Input((None, None, self.input_channels))

        #residual blocks.
        conv2d_residual = Conv2D(filters = 64,kernel_size = (3, 3), padding = "same", activation = "relu")(input_shape)
        for i in range(depth - 2):
            conv2d_residual = Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu")(conv2d_residual)

        conv2d_residual = Conv2D(filters = self.input_channels, kernel_size = (3, 3), padding = "same")(conv2d_residual)

        #skip connection
        skip_connection = Add()([input_shape, conv2d_residual])

        model = Model(inputs = input_shape, outputs = skip_connection)
        model.summary()

        return model

    def DRCN(self, recursive_depth = 16, filter_num = 128): 
        """
        recursive_depth : numbers of recursive_conv2d.
        filter_num : filter numbers.(default 256)
        """
        Inferencd_conv2d = Conv2D(filters = filter_num, kernel_size = (3, 3), padding = "same", activation = "relu")
        """
        Inferencd_conv2d : Inference net.(same weight)
        """
        #model
        #input single image
        input_shape = Input((None, None, self.input_channels))

        #Embedding net.
        conv2d_0 = Conv2D(filters = filter_num, kernel_size = (3, 3), padding = "same", activation = "relu")(input_shape)
        conv2d_1 = Conv2D(filters = filter_num, kernel_size = (3, 3), padding = "same", activation = "relu")(conv2d_0)      

        #Inference net and Reconstruction net.
        weight_list = recursive_depth * [None]
        pred_list = recursive_depth * [None]

        for i in range(recursive_depth):
            Inferencd_output = Inferencd_conv2d(conv2d_1)
            Recon_0 = Conv2D(filters = filter_num, kernel_size = (3, 3), padding = "same", activation = "relu")(Inferencd_output)
            Recon_1 = Conv2D(filters = self.input_channels, kernel_size = (3, 3), padding = "same", activation = "relu")(Recon_0)
            weight_list[i] = Recon_1
    
        for i in range(recursive_depth):
            skip_connection = Add()([weight_list[i], input_shape])
            pred = Multiply()([weight_list[i], skip_connection])

            pred_list[i] = pred

        pred = Add()(pred_list)

        model = Model(inputs = input_shape, outputs = pred)
        model.summary()

        return model

    def RED_Net(self, conv_num = 10, deconv_num = 10, filter_num = 64, kernel_size = (3, 3)):
        #create a list to store skip connections
        skip_connection_list = (conv_num // 2) * [None]

        input_shape = Input((None, None, self.input_channels))
        skip_connection_list[0] = input_shape
        
        #convolution part   
        conv = input_shape
        for i in range(conv_num):
            conv = Conv2D(filters = filter_num, kernel_size = kernel_size, padding = "same", activation = "relu")(conv)

            if i % 2 == 1 and i != conv_num - 1:
                skip_connection_list[(i // 2 + 1)] = conv

        #deconvolution part
        for i in range(deconv_num - 1):
            conv = Conv2DTranspose(filters = filter_num, kernel_size = kernel_size, padding = "same", activation = "relu")(conv)
            
            #add skip connections
            if i % 2 == 1:
                deconv_skip = Add()([conv, skip_connection_list[-1 * (i // 2 + 1)]])
                conv = deconv_skip

        conv = Conv2DTranspose(filters = self.input_channels, kernel_size = kernel_size, padding = "same", activation = "relu")(conv)
        deconv_skip = Add()([conv, skip_connection_list[0]])

        model = Model(inputs = input_shape, outputs = deconv_skip)
        model.summary()

        return model
        
class Video_SR():
    def __init__(self):
        self.upsampling_scale = 2
        self.input_channels = 1
        """
        upsampling_scale : magnification
        input_channels : channels of input_img.(gray → 1, RGB → 3)
        """

    #DeepSR
    def DeepSR(self, input_LR_num = 4):
        #input video frames
        input_list = input_LR_num * [None]
        for img in range(input_LR_num): 
            input_list[img] = Input(shape = (None, None, self.input_channels), name = "input_" + str(img))

        #concatenate
        input_shape = Concatenate()(input_list)

        #convolution
        conv2d_0 = Conv2D(filters = 256, kernel_size = (11, 11), padding = "same", activation = "tanh")(input_shape)
        conv2d_1 = Conv2D(filters = 512, kernel_size = (1, 1), padding = "same", activation = "tanh")(conv2d_0)
        conv2d_2 =  Conv2D(filters = 1, kernel_size = (3, 3), padding = "same", activation = "tanh")(conv2d_1)

        #deconvolution
        deconv2d_0 = Conv2DTranspose(filters = 1, kernel_size = (25, 25), strides = (1, 1), padding = "same")(conv2d_2)

        model = Model(inputs = input_list, outputs = deconv2d_0)
        model.summary()

        return model

    #VSRnet model b
    def VSRnet(self, input_LR_num = 3):
        #input video frames
        input_list = input_LR_num * [None]
        for img in range(input_LR_num): 
            input_list[img] = Input(shape = (None, None, self.input_channels), name = "input_" + str(img))

        #convolution each images
        conv2d_0_tminus1 = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_list[0])
        conv2d_0_t = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_list[1])
        conv2d_0_tplus1 = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_list[2])

        #concatenate each results
        new_input = Concatenate()([conv2d_0_tminus1, conv2d_0_t, conv2d_0_tplus1])

        #convolution
        conv2d_1 = Conv2D(filters = 32, kernel_size = (5, 5), padding = "same", activation = "relu")(new_input)
        conv2d_2 =  Conv2D(filters = 1, kernel_size = (5, 5), padding = "same")(conv2d_1)

        model = Model(inputs = input_list, outputs = conv2d_2)
        model.summary()

        return model

    #RVSR
    def RVSR(self, input_LR_num = 5):
        #ESPCN used in RVSR
        def RVSR_ESPCN(input_list, input_channels, mag):
            if len(input_list) == 1:
                input_shape = input_list[0]
            else:
                input_shape = Concatenate()(input_list)

            conv2d_0 = Conv2D(filters = len(input_list) * input_channels, kernel_size = (5, 5), padding = "same", activation = "relu")(input_shape)
            conv2d_1 = Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu")(conv2d_0)
            conv2d_2 = Conv2D(filters = mag ** 2, kernel_size = (3, 3), padding = "same")(conv2d_1)

            pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, mag))(conv2d_2)
        
            return pixel_shuffle

        input_list = input_LR_num * [None]
        output_list = (input_LR_num // 2 + 1) * [None]

        #input video frames
        for img in range(input_LR_num): 
            input_list[img] = Input(shape = (None, None, self.input_channels), name = "input_" + str(img))

        #create output images
        for num in range(0, input_LR_num // 2 + 1):
            output = RVSR_ESPCN(input_list[input_LR_num // 2 - num : input_LR_num // 2 + num + 1], self.input_channels, self.upsampling_scale)
            output_list[num] = output
        
        #average
        Tem_agg_model = Average()(output_list)

        model = Model(inputs = input_list, outputs = [Tem_agg_model])
        model.summary()

        return model

    #VESPCN
    def VESPCN(self, input_LR_num = 3): #main
        def Coarse_flow(input_list, upscale_factor):
            input_shape = Concatenate()(input_list)

            conv2d_0 = Conv2D(filters = 24, kernel_size = (5, 5), strides = (2, 2), padding = "same", activation = "relu")(input_shape)
            conv2d_1 = Conv2D(filters = 24, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu")(conv2d_0)
            conv2d_2 = Conv2D(filters = 24, kernel_size = (5, 5), strides = (2, 2), padding = "same", activation = "relu")(conv2d_1)
            conv2d_3 = Conv2D(filters = 24, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu")(conv2d_2)
            conv2d_4 = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "tanh")(conv2d_3)

            pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, upscale_factor))(conv2d_4)

            delta_x = Multiply()([input_list[1], pixel_shuffle])
            delta_x = Add()([tf.expand_dims(delta_x[:,:,:,0], -1), tf.expand_dims(delta_x[:,:,:,1], -1)])

            I_coarse = Add()([input_list[1], delta_x])

            return pixel_shuffle, I_coarse

        def Fine_flow(input_list, upscale_factor):
            input_shape = Concatenate()(input_list)

            conv2d_0 = Conv2D(filters = 24, kernel_size = (5, 5), strides = (2, 2), padding = "same", activation = "relu")(input_shape)
            conv2d_1 = Conv2D(filters = 24, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu")(conv2d_0)
            conv2d_2 = Conv2D(filters = 24, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu")(conv2d_1)
            conv2d_3 = Conv2D(filters = 24, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu")(conv2d_2)
            conv2d_4 = Conv2D(filters = 8, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "tanh")(conv2d_3)

            pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, upscale_factor))(conv2d_4)

            return pixel_shuffle   

        def MES(input_list): #Motion estimation
            delta_c, I_c = Coarse_flow(input_list, upscale_factor = 4)
    
            input_list.append(delta_c)
            input_list.append(I_c)

            delta_f = Fine_flow(input_list, upscale_factor = 2)

            delta = Add()([delta_c, delta_f])
            delta_x = Multiply()([input_list[1], delta])
            delta_x = Add()([tf.expand_dims(delta_x[:,:,:,0], -1), tf.expand_dims(delta_x[:,:,:,1], -1)])

            I_MES = Add()([input_list[1], delta_x])

            return I_MES

        def VESPCN_ESPCN(input_list, input_channels, mag):
            input_shape = Concatenate()(input_list)

            conv2d_0 = Conv2D(filters = len(input_list) * input_channels, kernel_size = (5, 5), padding = "same", activation = "relu")(input_shape)
            conv2d_1 = Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu")(conv2d_0)
            conv2d_2 = Conv2D(filters = mag ** 2, kernel_size = (3, 3), padding = "same")(conv2d_1)

            pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, mag))(conv2d_2)
        
            return pixel_shuffle

        #input video frames
        input_list = input_LR_num * [None]
        for img in range(input_LR_num): 
            input_list[img] = Input(shape = (None, None, self.input_channels), name = "input_" + str(img))

        I_t_minus_1 = MES([input_list[1], input_list[0]])
        I_t_plus_1 = MES([input_list[1], input_list[-1]])

        #upsampling
        ESPCN_input = [I_t_minus_1, input_list[0], I_t_plus_1]
        result_t = VESPCN_ESPCN(ESPCN_input, len(ESPCN_input), self.upsampling_scale)

        model = Model(inputs = input_list, outputs = result_t)

        model.summary()
        return model














