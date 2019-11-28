# coding:utf-8

import tensorflow as tf
import argparse
import os
import time
import pickle
import ipdb

from social_model import SocialModel
from social_utils import SocialDataLoader
from grid import getSequenceGridMask


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    # Type of recurrent unit parameter
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=16, #default 16
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=2,   #default 20
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=2,    #before the default was 50
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4, #default 4
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=70,
                        help='Maximum Number of Pedestrians')
    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=1,
                        help='The dataset index to be left out in training')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    args = parser.parse_args()
    train(args)


def train(args):
    datasets = range(5)
    # Remove the leaveDataset from datasets
    datasets.remove(args.leaveDataset)  #dataset=[1,2,3,4]

    # Create the SocialDataLoader object
    data_loader = SocialDataLoader(args.batch_size, args.seq_length, args.maxNumPeds, datasets, forcePreProcess=True, infer=False)

    # Log directory
    log_directory = 'log/'
    log_directory += str(args.leaveDataset) + '/'

    # Logging files
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    save_directory = 'save/'
    save_directory += str(args.leaveDataset) + '/'

    with open(os.path.join(save_directory, 'social_config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create a SocialModel object with the arguments
    model = SocialModel(args)



    config = tf.ConfigProto()
    #动态申请显存
    config.gpu_options.allow_growth=True

    #设置tf.ConfigProto()中参数log_device_placement = True ,可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,
    # 会在终端打印出各项操作是在哪个设备上运行的。
    config=tf.ConfigProto(log_device_placement=True) # Showing which device is allocated (in case of multiple GPUs)

    #限制GPU使用率
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # Allocating 20% of memory in each GPU with 0.5


    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Initialize all variables in the graph
        # Nella versione di tf<1.0 era : sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        # Initialize a saver that saves all the variables in the graph

        # Nella versione di tf<1.0 era : saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)  # max_to_keep: maximum number of trained model(parameter) are stored

        # summary_writer = tf.train.SummaryWriter('/tmp/lstm/logs', graph_def=sess.graph_def)
        print 'Training begin'
        best_val_loss = 100
        best_epoch = 0

        # For each epoch
        #会对一个epoch中所有的batch的batch_loss(也就是train_loss)计算总和并平均，
        for e in range(args.num_epochs):

            # Assign the learning rate value for this epoch
            # 学习率一直在下降
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))  #tf.assign(A, new_number) : assign new_number to A
            # Reset the data pointers in the data_loader
            data_loader.reset_batch_pointer(valid=False)   # Go to the first frame of the first (train？) dataset

            loss_epoch = 0

            # For each batch
            #每训练一个batch，输出一个train_loss = loss_batch
            for b in range(data_loader.num_batches):
                # Tic
                start = time.time()  #记录开始时刻的时间

                # Get the source, target and dataset data for the next batch
                # x, y are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # d is the list of dataset indices from which each batch is generated (used to differentiate between datasets)
                x, y, d = data_loader.next_batch()   #ATTENTION: y并不是y坐标，而是terget data  size: seq_length x maxNumPeds x 3

                # variable to store the loss for this batch
                loss_batch = 0

                # For each sequence in the batch
                for each_seq in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    x_batch, y_batch, d_batch = x[each_seq], y[each_seq], d[each_seq]

                    if d_batch == 0 and datasets[0] == 0:
                        dataset_data = [640, 480]  #QUESTION
                                                    #可能是因为图片大小不一样？  第0个dataset的图片要小一点？
                    else:
                        dataset_data = [720, 576]

                    #该序列下，由每帧一个的frame_mask组成的sequence_mask
                    grid_batch = getSequenceGridMask(x_batch, dataset_data, args.neighborhood_size, args.grid_size)

                    # Feed the source, target data
                    feed = {model.input_data: x_batch, model.target_data: y_batch, model.grid_data: grid_batch}

                    train_loss, _ = sess.run([model.cost, model.train_op], feed)    #QUESTION   what is _    ?
                                                                             #类似与matlab中的ans。即输出值不用赋给变量（我们不关心这个输出是什么）

                    loss_batch += train_loss

                end = time.time()
                loss_batch = loss_batch / data_loader.batch_size
                loss_epoch += loss_batch
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"  #显示现在处于哪个batch
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        loss_batch, end - start))

                # Save the model if the current epoch and batch number match the frequency
                '''
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join('save', 'social_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                '''


            loss_epoch /= data_loader.num_batches
            log_file_curve.write(str(e)+','+str(loss_epoch)+',')  #report loss_epoch in each epoch
            print '*****************'

            # Validation
            data_loader.reset_batch_pointer(valid=True)
            loss_epoch = 0

            #计算model在validation dataset中的loss
            for b in range(data_loader.num_batches):

                # Get the source, target and dataset data for the next batch
                # x, y are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # d is the list of dataset indices from which each batch is generated (used to differentiate between datasets)
                x, y, d = data_loader.next_valid_batch()

                # variable to store the loss for this batch
                loss_batch = 0

                # For each sequence in the batch
                for each_seq in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    x_batch, y_batch, d_batch = x[each_seq], y[each_seq], d[each_seq]

                    if d_batch == 0 and datasets[0] == 0:
                        dataset_data = [640, 480]
                    else:
                        dataset_data = [720, 576]

                    grid_batch = getSequenceGridMask(x_batch, dataset_data, args.neighborhood_size, args.grid_size)

                    # Feed the source, target data
                    feed = {model.input_data: x_batch, model.target_data: y_batch, model.grid_data: grid_batch}

                    train_loss = sess.run(model.cost, feed)

                    loss_batch += train_loss

                loss_batch = loss_batch / data_loader.batch_size
                loss_epoch += loss_batch

            loss_epoch /= data_loader.valid_num_batches

            # Update best validation loss until now
            if loss_epoch < best_val_loss:
                best_val_loss = loss_epoch
                best_epoch = e

            #找出在哪个epoch时的model最理想
            print('(epoch {}), valid_loss = {:.3f}'.format(e, loss_epoch))
            print 'Best epoch', best_epoch, 'Best validation loss', best_val_loss  #find the best epoch till now
            log_file_curve.write(str(loss_epoch)+'\n')
            print '*****************'

            # Save the model after each epoch
            print 'Saving model'
            checkpoint_path = os.path.join(save_directory, 'social_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=e)    #store the trained model
                                         # global_step=e : Add the number of trainings as a suffix to the model name

            print("model saved to {}".format(checkpoint_path))

        print 'Best epoch', best_epoch, 'Best validation loss', best_val_loss
        log_file.write(str(best_epoch)+','+str(best_val_loss))

        # CLose logging files
        log_file.close()         #only report the best epoch and its best_val_loss
        log_file_curve.close()   # #report loss_epoch in each epoch


if __name__ == '__main__':
    main()
