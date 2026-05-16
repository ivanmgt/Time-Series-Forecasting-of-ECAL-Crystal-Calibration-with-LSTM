# This script is used to prepare the dataset we need
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams.update({'font.size': 17})


# This class is used to generate the ECAL-Dataset
# And its input is a csv file name
class ECAL_Dataset_Prep:
    def __init__(self, csv_file, input_len, output_len, stride, fig_name_cali, fig_name_scaled_cali, verbose=False, plt_show=True):
        self.csv_file = csv_file
        self.il = input_len
        self.ol = output_len
        self.stride = stride  # to ensure there is no overlap in prediction, please set the stride = output_window
        self.fig_name_cali = fig_name_cali
        self.fig_name_scaled_cali = fig_name_scaled_cali
        self.verbose = verbose # True: print information; False: will not print information
        self.plt_show = plt_show # True: show plot; False: will not show plot

        # the scalers
        self.scaler_cali = StandardScaler()
        self.scaler_lumi = StandardScaler()
        # the dataframe version
        self.df_cali = None
        self.df_lumi = None
        # the numpy version
        self.np_cali = None
        self.np_lumi = None
        # the tensor/torch version
        self.torch_cali = None
        self.torch_lumi = None
        # the numpy version of input and target samples
        self.np_X = None
        self.np_Y = None
        # the tensor/torch version of input and target samples
        self.torch_X = None
        self.torch_Y = None

    def start_processing(self):
        # now, we will call functions one by one to update the values
        self.get_df()
        self.normalize_dataset()
        self.sequence_dataset()
        self.visualize_data_samples()

    # get the dataframe and split the dataset into two dataframes:
    # one for calibration
    # one for luminosity diff
    def get_df(self):
        self.df = pd.read_csv(self.csv_file, index_col=0)
        self.df.index = pd.to_datetime(self.df['laser_datetime'])
        try:
            self.df_cali = self.df[['calibration']].copy()
            self.df_lumi = self.df[['delta_lumi']].copy()
        except:
            assert False, "We except the csv should at least include ['calibration', 'delta_lumi'] (even they are empty columns)!"

    # normalize the dataset and also get other numpy and torch version
    def normalize_dataset(self):
        # normalize calibration
        if len(self.df_cali) !=0:
            self.scaler_cali.fit(self.df_cali[['calibration']])
            self.df_cali['calibration_scaled'] = None
            self.df_cali.loc[:,'calibration_scaled'] = self.scaler_cali.transform(self.df_cali[['calibration']])
            if self.verbose:
                print(self.df_cali.describe())
            self.np_cali = self.df_cali['calibration_scaled'].to_numpy()
            self.np_cali = self.np_cali.reshape(-1, 1)

        # normalize luminosity diff
        if len(self.df_lumi) !=0:
            self.scaler_lumi.fit(self.df_lumi[['delta_lumi']])
            self.df_lumi['delta_lumi_scaled'] = None
            self.df_lumi.loc[:,'delta_lumi_scaled'] = self.scaler_lumi.transform(self.df_lumi[['delta_lumi']])
            if self.verbose:
                print(self.df_lumi.describe())
            self.np_lumi = self.df_lumi['delta_lumi_scaled'].to_numpy()
            self.np_lumi = self.np_lumi.reshape(-1, 1)

    def sequence_dataset(self):
        # please note that: we arrange features in this order:
        # the first feature is lumi_diff, which is the "luminosity delta"
        # the second feature is cali, which is the "calibration"
        num_lumi = self.np_lumi.shape[0]
        num_cali = self.np_cali.shape[0]

        #case: we have the same number of cali & lumi
        if num_lumi == num_cali and num_lumi!=0:
            num_samples = (num_lumi- self.il - self.ol) // self.stride + 1
            #here, we want to combine the luminosity and calibration as our input
            y1_combined = np.hstack((self.np_lumi, self.np_cali))
            num_features_combined = y1_combined.shape[1]
            X = np.zeros([self.il, num_samples, num_features_combined])
            Y = np.zeros([self.ol, num_samples, num_features_combined])

            #processing X---input samples
            for ii in np.arange(num_samples):
                start_x = self.stride * ii
                end_x = start_x + self.il
                X[:, ii, :] = y1_combined[start_x:end_x, :]
            self.np_X = X
            self.torch_X = torch.from_numpy(X).type(torch.Tensor)

            ### processing Y---target samples
            for ii in np.arange(num_samples):
                start_y = self.stride * ii + self.il
                end_y = start_y + self.ol
                Y[:, ii, :] = y1_combined[start_y:end_y, :]
            self.np_Y = Y
            self.torch_Y = torch.from_numpy(Y).type(torch.Tensor)

        elif num_lumi>0 and num_cali==0:
            num_samples = (num_lumi- self.il - self.ol) // self.stride + 1
            Y = np.zeros([self.ol, num_samples, 1])
            ### processing Y---target samples
            for ii in np.arange(num_samples):
                start_y = self.stride * ii + self.il
                end_y = start_y + self.ol
                Y[:, ii, :] = self.np_lumi[start_y:end_y, :]
            self.np_Y = Y
            self.torch_Y = torch.from_numpy(Y).type(torch.Tensor)

        else:
            assert False, 'We only support two cases: 1) Cali and Lumi. have the same length; 2) We only have Lumi.!'


    def visualize_data_samples(self):
        plt.figure(figsize=(18, 6))
        plt.plot(self.df_cali.index, self.df_cali['calibration'], color='k', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Calibration')
        plt.title(
            'Before normalization: Mean={}; Std={}'.format(round( self.df_cali['calibration'].mean(), 3), round( self.df_cali['calibration'].std(), 3)))
        plt.savefig(self.fig_name_cali, dpi=300)
        if self.plt_show:
            plt.show()
        plt.close()

        plt.figure(figsize=(18, 6))
        plt.plot(self.df_cali.index, self.df_cali['calibration_scaled'], color='k', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Calibration')
        plt.title('After normalization: Mean={}; Std={}'.format(round(self.df_cali['calibration_scaled'].mean(), 3),
                                           round(self.df_cali['calibration_scaled'].std(), 3)))
        plt.savefig(self.fig_name_scaled_cali, dpi=300)
        if self.plt_show:
            plt.show()
        plt.close()
        
# This script includes additional helper functions we need
import torch
import matplotlib.pyplot as plt

def save_model(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)
    print('The trained model has been saved!')

def plot_loss(loss, fig_name):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.close()

def show_loss(loss,plt_show=False):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training loss vs. Epoch')
    plt.tight_layout()
    if plt_show:
        plt.show()
    plt.close()

if __name__ == '__main__':
    pass

### This script defines the seq2seq model we are using
### We are using LSTM for Encoder & Decoder
import torch.nn as nn

class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2):
        # Define LSTM-Encoder,
        # which will encode the time-series sequence to a latent code
        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        return lstm_out, self.hidden     
    

class LSTM_Decoder(nn.Module):
    # Define LSTM-Decoder,
    # which will decode the latent code/hidden state generated by the LSTM-Encoder
    # Decodes hidden state output by encoder
    
    def __init__(self, input_size, hidden_size, num_layers = 2):
        super(LSTM_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers)

        # here our output is only calibration, thus why the output dim is 1
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))
        return output, self.hidden
    
# This script includes the functions to make predictions on data


class Seq2Seq_Prediction:
    def __init__(self,
                encoder,
                decoder,
                Xtrain,
                Ytrain,
                df,
                scaler_cali,
                device,
                fig_name_mape,
                fig_name_mse,
                metric_file,
                strategy,
                plt_show=True):

        self.encoder = encoder
        self.decoder = decoder
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.df = df
        self.scaler_cali = scaler_cali
        self.device = device
        self.fig_name_mape = fig_name_mape
        self.fig_name_mse = fig_name_mse
        self.metric_file = metric_file
        self.strategy = strategy
        self.plt_show = plt_show # True: show plot; False: will not show plot

    def start_prediction(self):

        if self.strategy == 'case1':  # do not use prediction as input to help the next-round prediction
            print('>>> ', self.strategy, ': start prediction...(be patient)')
            self.prediction_case1()
            print('>>> Finish prediction!')

        elif self.strategy == 'case2':  # use prediction as input to help the next-round prediction
            print('>>> ', self.strategy, ': start prediction...(be patient)')
            self.prediction_case2()
            print('>>> Finish prediction!')

        else:
            assert False, "Please select one of them---[case1, case2]!"

    def getAPE(self):
        return [self.meanAPE,self.length]

    # case1: we do not use prediction as input to help next-round prediction
    def prediction_case1(self):

        # get the correct scaler
        # if 'train' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['train']
        # elif 'val' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['val']
        # elif 'test' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['test']

        iw = self.Xtrain.shape[0]
        ow = self.Ytrain.shape[0]

        batches = self.Xtrain.shape[1]

        pred_Ytrain = np.zeros((self.Ytrain.shape[0], self.Ytrain.shape[1], 1))

        # plot training/test predictions
        for ii in range(batches):
            # train set
            X_train_temp = self.Xtrain[:, ii, :]
            Y_train_temp = self.Ytrain[:, ii, :]
            input_tensor = torch.from_numpy(X_train_temp).type(torch.Tensor).to(self.device)
            target_tensor = torch.from_numpy(Y_train_temp).type(torch.Tensor).to(self.device)
            Y_train_pred = self.predict(self.encoder, self.decoder, input_tensor, target_tensor, target_len=ow)
            pred_Ytrain[:, ii:ii + 1, :] = Y_train_pred

        ### after we have all the predictions, we want to plot the fig and compute its performance metric
        ### first, we want to flatten our windowed data
        Ytrain = self.Ytrain[:, :, 1:]
        GT_np = self.from_seq2norm(Ytrain)
        GT_np = np.asarray(GT_np).reshape(-1, 1)
        GT_np_org_scale = self.scaler_cali.inverse_transform(GT_np)
        Pred_np = self.from_seq2norm(pred_Ytrain)
        Pred_np = np.asarray(Pred_np).reshape(-1, 1)
        Pred_np_org_scale = self.scaler_cali.inverse_transform(Pred_np)

        meanAPE = self.MAPE_Metric(GT_np_org_scale, Pred_np_org_scale)
        fig_title = 'MAPE = {}'.format(meanAPE)
        T_info = self.df.index[iw:iw + len(GT_np_org_scale)]
        lumi_info = self.df['delta_lumi'][iw:iw + len(GT_np_org_scale)]
        self.plot_prediction(GT_np_org_scale, Pred_np_org_scale, lumi_info, T_info, self.fig_name_mape, fig_title)
        # plot_prediction(GT_np, GT_np, lumi_info, T_info, fig_name, fig_title)

        mse = self.MSE_Metric(GT_np_org_scale, Pred_np_org_scale)
        # fig_title = 'MSE = {}'.format(mse)
        # T_info = self.df.index[iw:iw + len(GT_np_org_scale)]
        # lumi_info = self.df['delta_lumi'][iw:iw + len(GT_np_org_scale)]
        # self.plot_prediction(GT_np_org_scale, Pred_np_org_scale, lumi_info, T_info, self.fig_name_mse, fig_title)

        # now, we want to save the metrics into the metric file
        metric_dict = {'MAP': [meanAPE], 'MSE': [mse]}
        metric_df = pd.DataFrame.from_dict(metric_dict)
        metric_df.to_csv(self.metric_file, index=False)
        return


    # case2: we use prediction as input to help next-round prediction
    def prediction_case2(self):

        # get the correct scaler
        # if 'train' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['train']
        # elif 'val' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['val']
        # elif 'test' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['test']

        iw = self.Xtrain.shape[0]
        ow = self.Ytrain.shape[0]

        batches = self.Xtrain.shape[1]

        pred_Ytrain = np.zeros((self.Ytrain.shape[0], self.Ytrain.shape[1], 1))

        Y_train_pred = []
        for ii in range(batches):
            # train set
            if ii == 0:
                X_train_temp = self.Xtrain[:, ii, :]
            else:
                X_train_temp = self.Xtrain[:, ii, :]
                X_train_temp[:, 1] = Y_train_pred.reshape(-1)
            Y_train_plt = self.Ytrain[:, ii, :]
            input_tensor = torch.from_numpy(X_train_temp).type(torch.Tensor).to(self.device)
            target_tensor = torch.from_numpy(Y_train_plt).type(torch.Tensor).to(self.device)
            Y_train_pred = self.predict(self.encoder, self.decoder, input_tensor, target_tensor, target_len=ow)
            pred_Ytrain[:, ii:ii + 1, :] = Y_train_pred

        ### after we have all the predictions, we want to plot the fig and compute its performance metric
        ### first, we want to flatten our windowed data
        Ytrain = self.Ytrain[:, :, 1:]
        GT_np = self.from_seq2norm(Ytrain)
        GT_np = np.asarray(GT_np).reshape(-1, 1)
        GT_np_org_scale = self.scaler_cali.inverse_transform(GT_np)
        Pred_np = self.from_seq2norm(pred_Ytrain)
        Pred_np = np.asarray(Pred_np).reshape(-1, 1)
        Pred_np_org_scale = self.scaler_cali.inverse_transform(Pred_np)

        meanAPE = self.MAPE_Metric(GT_np_org_scale, Pred_np_org_scale)
        fig_title = 'MAPE = {}'.format(meanAPE)
        T_info = self.df.index[iw:iw + len(GT_np_org_scale)]
        lumi_info = self.df['delta_lumi'][iw:iw + len(GT_np_org_scale)]
        self.plot_prediction(GT_np_org_scale, Pred_np_org_scale, lumi_info, T_info, self.fig_name_mape, fig_title)

        mse = self.MSE_Metric(GT_np_org_scale, Pred_np_org_scale)
        # fig_title = 'MSE = {}'.format(mse)
        # T_info = self.df.index[iw:iw + len(GT_np_org_scale)]
        # lumi_info = self.df['delta_lumi'][iw:iw + len(GT_np_org_scale)]
        # self.plot_prediction(GT_np_org_scale, Pred_np_org_scale, lumi_info, T_info, self.fig_name_mse, fig_title)

        # now, we want to save the metrics into the metric file
        metric_dict = {'MAP': [meanAPE], 'MSE': [mse]}
        metric_df = pd.DataFrame.from_dict(metric_dict)
        metric_df.to_csv(self.metric_file, index=False)

        return

    def predict(self, encoder, decoder, input_tensor, target_tensor, target_len):
        # This function is used to make prediction once the model is trained
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            # extend the input tensor to correct dim
            input_tensor = input_tensor.unsqueeze(1)
            target_tensor_2features = target_tensor.unsqueeze(1)

            # target_tensor_input is the "luminosity delta", which will be used as input to the decoder
            target_tensor_input = target_tensor_2features[:, :, 0:1]

            # target_tensor is the "calibration", which is the value we want to predict
            target_tensor = target_tensor_2features[:, :, 1:]

            encoder_output, encoder_hidden = encoder(input_tensor)

            # initialize tensor for predictions
            outputs = torch.zeros(target_len, target_tensor.shape[2])

            # decode input_tensor
            decoder_input = input_tensor[-1, :, :]  # the initialization, can be any values
            decoder_hidden = encoder_hidden

            # make prediction step by step
            for t in range(target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output.squeeze(0)
                lumi_feature = target_tensor_input[t, :, :]
                decoder_output = torch.cat((lumi_feature, decoder_output), dim=1)
                decoder_input = decoder_output

            np_outputs = outputs.detach().unsqueeze(1)
            np_outputs = np_outputs.numpy()

        return np_outputs

    # This function converts the seq to normal format
    def from_seq2norm(self, input_np):
        result = []

        total_batch_num = input_np.shape[1]

        for cur_b in range(total_batch_num):
            cur_batch_data = input_np[:, cur_b:cur_b + 1, :]
            sample_num = cur_batch_data.shape[1]
            for cur_idx in range(sample_num):
                result.extend((cur_batch_data[:, cur_idx, :]).flatten().tolist())
        return result

    def plot_prediction(self, target, pred, lumi_info, time_info, fig_name, fig_title):
        #### double Y figure
        fig, ax1 = plt.subplots(figsize=(16, 9))  # fig, ax1 = plt.subplots(figsize=(25, 5))
        plt.title(fig_title, fontsize=20)
        plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=14)
        plot1 = ax1.plot(time_info, target, color='b', linewidth=2, label='Calibration (true)')
        plot2 = ax1.plot(time_info, pred, color='r', linewidth=2, label='Calibration (prediction)')
        ax1.set_ylabel('Calibration', fontsize=18)
        ax1.yaxis.label.set_color('b')
        ax1.set_xlabel('Time Info', fontsize=18)
        ax1.set_ylim(0.7, 1)
        plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ### now, start the other y plot
        ax2 = ax1.twinx()
        plot3 = ax2.plot(time_info, lumi_info, label='Luminosity', color='grey', linewidth=1, linestyle='dashed')
        ax2.set_ylabel('Luminosity', fontsize=18)
        ax2.yaxis.label.set_color('grey')
        # ax2.set_ylim(0, 0.08)
        # ax2.set_xlim(1966, 2014.15)
        # ax2.tick_params(axis='y', labelsize=14)
        for tl in ax2.get_yticklabels():
            tl.set_color('grey')

        lines = plot1 + plot2 + plot3
        ax1.legend(lines, [l.get_label() for l in lines])
        plt.tight_layout()
        plt.savefig(fig_name, dpi=300)
        if self.plt_show:
            plt.show()
        plt.close()

    def MAPE_Metric(self, GT_np, Pred_np):
        if len(GT_np) != len(Pred_np):
            assert False, 'GT_np and Pred_np must have the same length!'
        APES = []
        for i in range(len(GT_np)):
            ape = abs((Pred_np[i] - GT_np[i]) / (GT_np[i]))
            if np.isnan(ape):
                continue
            APES.append(ape)
        meanAPE = (sum(APES) * 100 / len(APES))
        meanAPE = np.round(meanAPE, 3)[0]
        self.length = len(APES)
        self.meanAPE = meanAPE
        return meanAPE

    def MSE_Metric(self, GT_np, Pred_np):
        if len(GT_np) != len(Pred_np):
            assert False, 'GT_np and Pred_np must have the same length!'

        GT_np_arr = np.asarray(GT_np)
        Pred_np_arr = np.asarray(Pred_np)
        sum_pow = np.sum(np.power(GT_np_arr - Pred_np_arr, 2))
        # sqrt_sum = np.sqrt(sum_pow)
        mse = sum_pow / len(GT_np)
        return mse








if __name__ == '__main__':
    pass

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random


class Seq2Seq_Train:
    def __init__(self,
                encoder,
                decoder,
                input_tensor,
                target_tensor,
                n_epochs,
                target_len,
                batch_size,
                learning_rate=0.01,
                opt_alg='adam',
                print_step=1,
                strategy = 'recursive',
                teacher_forcing_ratio=0.5,
                device='cpu',
                loss_figure_name='loss.png',
                verbose=True,
                plt_show=False):

        self.encoder = encoder
        self.decoder = decoder
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor
        self.n_epochs = n_epochs
        self.target_len = target_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.opt_alg = opt_alg
        self.print_step = print_step
        self.strategy = strategy
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.loss_figure_name = loss_figure_name
        self.verbose = verbose # True: print information; False: will not print information
        self.plt_show = plt_show # True: show plot; False: will not show plot

    def start_train(self):
        print('>>> Start training... (be patient: training time varies)')
        if self.strategy == 'recursive':
            self.train_model_recursive()

        elif self.strategy == 'teacher_forcing':
            self.train_model_teacher_forcing()

        elif self.strategy == 'mixed':
            self.train_model_mixed()

        else:
            assert False, "Please select one of them---[recursive, teacher_forcing, mixed]!"

        print('>>> Finish training!')

    def train_model_recursive(self):

        ### move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        ### get the learnable parameters
        params = []
        params += [x for x in self.encoder.parameters()]
        params += [x for x in self.decoder.parameters()]

        ### define optimizer method
        if self.opt_alg.upper() == 'ADAM':
            optimizer = optim.Adam(params=params, lr=self.learning_rate)
        elif self.opt_alg.upper() == 'SGD':
            optimizer = optim.SGD(params=params, lr=self.learning_rate)
        else:
            assert False, 'This version only supports ADAM and SGD!'

        ### define loss function
        criterion = nn.MSELoss()

        ### calculate number of batch iterations
        n_samples = self.input_tensor.shape[1]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        ### save loss
        losses = []

        for epoch in range(self.n_epochs):
            if self.verbose:
                print("======== epoch {} out of {} epochs ========".format(epoch, self.n_epochs))
            self.encoder.train()
            self.decoder.train()
            batch_loss = []

            for batch in range(n_batches):
                start = batch * self.batch_size
                end = min(start + self.batch_size, n_samples)
                if self.verbose:
                    print("batch {} out of {} batches (samples {}–{})".format(batch, n_batches, start, end))

        # select data
                input_batch = self.input_tensor[:, start:end, :]
                target_batch_input = self.target_tensor[:, start:end, 0:1]
                target_batch = self.target_tensor[:, start:end, 1:]

                # outputs tensor
                current_batch_size = end - start
                outputs = torch.zeros(self.target_len, current_batch_size, target_batch.shape[2])


                # initialize hidden state
                # encoder_hidden = self.encoder.init_hidden(batch_size)

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(input_batch)

                # decoder with teacher forcing
                decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden

                # different training strategies
                # predict recursively
                # make prediction step by step
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output
                    ### adding the other features
                    lumi_feature = target_batch_input[t, :, :]
                    decoder_output = torch.cat((lumi_feature, decoder_output), dim=1)
                    decoder_input = decoder_output

                # compute the loss
                outputs = outputs.to(self.device)
                loss = criterion(outputs, target_batch)
                batch_loss.append(loss.item())

                # backpropagation
                loss.backward()
                optimizer.step()
            epoch_loss = np.mean(batch_loss)
            #print('>>>>>> {}/{} Epoch; Loss={}'.format(epoch, self.n_epochs, epoch_loss))
            losses.append(epoch_loss)

            ### we save its loss every print_step
            if epoch % self.print_step == 0:
                plot_loss(losses, self.loss_figure_name)
        show_loss(losses,self.plt_show)

    def train_model_teacher_forcing(self):
        ### move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        ### get the learnable parameters
        params = []
        params += [x for x in self.encoder.parameters()]
        params += [x for x in self.decoder.parameters()]

        ### define optimizer method
        if self.opt_alg.upper() == 'ADAM':
            optimizer = optim.Adam(params=params, lr=self.learning_rate)
        elif self.opt_alg.upper() == 'SGD':
            optimizer = optim.SGD(params=params, lr=self.learning_rate)
        else:
            assert False, 'This version only supports ADAM and SGD!'

        ### define loss function
        criterion = nn.MSELoss()

        ### calculate number of batch iterations
        n_batches = int(self.input_tensor.shape[1] / self.batch_size)

        ### save loss
        losses = []

        for epoch in range(self.n_epochs):
            if self.verbose:
                print("======== epoch {} out of {} epochs ========".format(epoch,self.n_epochs))
            self.encoder.train()
            self.decoder.train()
            batch_loss = []

            for batch in range(n_batches):
                if self.verbose:
                    print("batch {} out of {} batches".format(batch,n_batches))
                # select data
                input_batch = self.input_tensor[:, batch: batch + self.batch_size, :]

                # target_batch_input means the "luminosity delta", which is given to us
                # we will combine this information with "calibration" as input to decoder each time
                target_batch_input = self.target_tensor[:, batch: batch + self.batch_size, 0:1]

                # target_batch means the "calibration", which is the value we want to predict
                # so target_batch is the real target
                target_batch = self.target_tensor[:, batch: batch + self.batch_size, 1:]

                # move data to device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                target_batch_input = target_batch_input.to(self.device)

                # outputs tensor
                outputs = torch.zeros(self.target_len, self.batch_size, target_batch.shape[2])

                # initialize hidden state
                # encoder_hidden = self.encoder.init_hidden(batch_size)

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(input_batch)

                # decoder with teacher forcing
                decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden

                # different training strategies
                # use teacher forcing
                if random.random() < self.teacher_forcing_ratio:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        decoder_input = target_batch[t, :, :]
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        decoder_input = torch.cat((lumi_feature, decoder_input), dim=1)

                # predict recursively
                else:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        decoder_output = torch.cat((lumi_feature, decoder_output), dim=1)
                        decoder_input = decoder_output

                # compute the loss
                outputs = outputs.to(self.device)
                loss = criterion(outputs, target_batch)
                batch_loss.append(loss.item())

                # backpropagation
                loss.backward()
                optimizer.step()
            epoch_loss = np.mean(batch_loss)
            #print('>>>>>> {}/{} Epoch; Loss={}'.format(epoch, self.n_epochs, epoch_loss))
            losses.append(epoch_loss)

            ### we save its loss every print_step
            if epoch % self.print_step == 0:
                plot_loss(losses, self.loss_figure_name)
        show_loss(losses,self.plt_show)

    def train_model_mixed(self):

        ### move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        ### get the learnable parameters
        params = []
        params += [x for x in self.encoder.parameters()]
        params += [x for x in self.decoder.parameters()]

        ### define optimizer method
        if self.opt_alg.upper() == 'ADAM':
            optimizer = optim.Adam(params=params, lr=self.learning_rate)
        elif self.opt_alg.upper() == 'SGD':
            optimizer = optim.SGD(params=params, lr=self.learning_rate)
        else:
            assert False, 'This version only supports ADAM and SGD!'

        ### define loss function
        criterion = nn.MSELoss()

        ### calculate number of batch iterations
        n_batches = int(self.input_tensor.shape[1] / self.batch_size)

        ### save loss
        losses = []

        for epoch in range(self.n_epochs):
            if self.verbose:
                print("======== epoch {} out of {} epochs ========".format(epoch,self.n_epochs))
            self.encoder.train()
            self.decoder.train()
            batch_loss = []

            for batch in range(n_batches):
                if self.verbose:
                    print("batch {} out of {} batches".format(batch,n_batches))
                # select data
                input_batch = self.input_tensor[:, batch: batch + self.batch_size, :]

                # target_batch_input means the "luminosity delta", which is given to us
                # we will combine this information with "calibration" as input to decoder each time
                target_batch_input = self.target_tensor[:, batch: batch + self.batch_size, 0:1]

                # target_batch means the "calibration", which is the value we want to predict
                # so target_batch is the real target
                target_batch = self.target_tensor[:, batch: batch + self.batch_size, 1:]

                # move data to device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                target_batch_input = target_batch_input.to(self.device)

                # outputs tensor
                outputs = torch.zeros(self.target_len, self.batch_size, target_batch.shape[2])
#plt.show
                # initialize hidden state
                # encoder_hidden = self.encoder.init_hidden(batch_size)

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(input_batch)

                # decoder with teacher forcing
                decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden

                # different training strategies
                # predict using mixed teacher forcing
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output

                    # predict with teacher forcing
                    if random.random() < self.teacher_forcing_ratio:
                        decoder_input = target_batch[t, :, :]
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        decoder_input = torch.cat((lumi_feature, decoder_input), dim=1)
                    # predict recursively
                    else:
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        decoder_output = torch.cat((lumi_feature, decoder_output), dim=1)
                        decoder_input = decoder_output

                # compute the loss
                outputs = outputs.to(self.device)
                loss = criterion(outputs, target_batch)
                batch_loss.append(loss.item())

                # backpropagation
                loss.backward()
                optimizer.step()
            epoch_loss = np.mean(batch_loss)
            #print('>>>>>> {}/{} Epoch; Loss={}'.format(epoch, self.n_epochs, epoch_loss))
            losses.append(epoch_loss)

            ### we save its loss every print_step
            if epoch % self.print_step == 0:
                plot_loss(losses, self.loss_figure_name)
        show_loss(losses,self.plt_show)
        
learning_rate = 1e-3
n_epochs = 200
print_step = 1
opt_alg = 'adam'
train_strategy = 'recursive' #"Please select one of them---[recursive, teacher_forcing, mixed]!"
teacher_forcing_ratio = 0.5 # please set it in the range of [0,1]
gpu_id = 0
num_layers = 2
verbose = False
plt_show = False
training_year = 2018 # model trained on this year [2016,2017]
device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

crystal_id = 30600
df_1=pd.read_csv("/data/plus_z_1/ring_1.csv")
base_dir = f'/results/seq2seq_mod_2018{crystal_id}' 
lista1=[12]
lista2=[64]
for i in lista1:
    for j in lista2:
        batch_size = 128
        input_len = i
        output_len = i
        stride = output_len
        if output_len>=48: batch_size =32
        hidden_size =j
        folder_name = 'LSTM_{}_IW_{}_OW_{}_LR_{}_ID_{}_train_year_{}'.format(hidden_size, input_len, output_len, learning_rate,crystal_id,training_year)
        # folder to save figures
        save_dir_vis_data =  os.path.join(base_dir,'{}/vis_data/'.format(folder_name))

        # folder to save models
        save_dir_models =  os.path.join(base_dir,'{}/models/'.format(folder_name))

        # folders for case1
        save_dir_case1_fig=  os.path.join(base_dir,'{}/case1_fig/'.format(folder_name))
        save_dir_case1_csv=  os.path.join(base_dir,'{}/case1_csv/'.format(folder_name))


      
        dir_list = [save_dir_vis_data, save_dir_case1_fig, save_dir_case1_csv,save_dir_models]
        for cur_dir in dir_list:
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)
                print('>>> {} has been created successfully!'.format(cur_dir))
            else:
                print('>>> {} is exist!'.format(cur_dir))
                 
         # for train_file_2016/2017

        X_train_all = None
        Y_train_all = None
        fig_name_cali = os.path.join(save_dir_vis_data, '{}_cali_original_ID_{}.png'.format(training_year,crystal_id))
        fig_name_scaled_cali = os.path.join(save_dir_vis_data, '{}_cali_scaled_ID_{}.png'.format(training_year,crystal_id))
        if training_year == 2018:
            train_file_2016=df_1[df_1["xtal_id"]==crystal_id].copy()
            train_file_2016["laser_datetime"]=pd.to_datetime(train_file_2016["laser_datetime"])
            train_file_2016=train_file_2016[train_file_2016["laser_datetime"].dt.year==2016].reset_index(drop=True)
            train_file_2016=train_file_2016.sort_values(by=["laser_datetime"]).reset_index(drop=True)
            train_file_2016["delta_lumi"]=train_file_2016["int_deliv_inv_ub"].diff()
            train_file_2016.loc[0,"delta_lumi"]=0
            csv_path = os.path.join(save_dir_vis_data, f"train_2016_ID_{crystal_id}.csv")
            train_file_2016.to_csv(csv_path, index=False)
            ecal_dataset_prep_train_2016 = ECAL_Dataset_Prep(csv_path, 
                                                            input_len, 
                                                            output_len, 
                                                            stride, 
                                                            fig_name_cali, 
                                                            fig_name_scaled_cali,
                                                            verbose=False,
                                                            plt_show=False)
            ecal_dataset_prep_train_2016.start_processing()

            X_train = ecal_dataset_prep_train_2016.torch_X
            Y_train = ecal_dataset_prep_train_2016.torch_Y
            
            os.remove(csv_path)
        elif training_year == 2017:
            test_file_2017=df_1[df_1["xtal_id"]==crystal_id].copy()
            test_file_2017["laser_datetime"]=pd.to_datetime(test_file_2017["laser_datetime"])
            test_file_2017=test_file_2017[test_file_2017["laser_datetime"].dt.year==2017].reset_index(drop=True)
            test_file_2017["delta_lumi"]=test_file_2017["int_deliv_inv_ub"].diff()
            test_file_2017.loc[0,"delta_lumi"]=0
            csv_path = os.path.join(save_dir_vis_data, f"train_2017_ID_{crystal_id}.csv")
            test_file_2017.to_csv(csv_path, index=False)
            ecal_dataset_prep_train_2017 = ECAL_Dataset_Prep(csv_path, 
                                                            input_len, 
                                                            output_len, 
                                                            stride, 
                                                            fig_name_cali, 
                                                            fig_name_scaled_cali,
                                                            verbose=False,
                                                            plt_show=False)
            ecal_dataset_prep_train_2017.start_processing()

            X_train = ecal_dataset_prep_train_2017.torch_X
            Y_train = ecal_dataset_prep_train_2017.torch_Y
            os.remove(csv_path)
        else:
            print("please use 2016 or 2017 to train the model")

        if X_train_all == None:
            X_train_all = X_train
            Y_train_all = Y_train
        else:
            X_train_all = torch.cat( (X_train_all,X_train), dim=1 )
            Y_train_all = torch.cat( (Y_train_all,Y_train), dim=1 )
        lstm_encoder = LSTM_Encoder(input_size=X_train_all.shape[2], hidden_size=hidden_size, num_layers=num_layers)
        lstm_decoder = LSTM_Decoder(input_size=Y_train_all.shape[2], hidden_size=hidden_size, num_layers=num_layers)
        lstm_encoder.to(device)
        lstm_decoder.to(device)
        print(lstm_encoder)
        print(lstm_decoder)
        loss_figure_name = os.path.join(save_dir_vis_data, '0_loss.png')
        target_len = output_len
        seq2seq_train = Seq2Seq_Train(lstm_encoder,
                                      lstm_decoder,
                                      X_train_all,
                                      Y_train_all,
                                      n_epochs,
                                      target_len,
                                      batch_size,
                                      learning_rate,
                                      opt_alg,
                                      print_step,
                                      train_strategy,
                                      teacher_forcing_ratio,
                                      device,
                                      loss_figure_name,
                                      verbose=False,
                                      plt_show=False)
        seq2seq_train.start_train()
        
        model_file_name = os.path.join(save_dir_models, 'lstm_encoder.pt')
        save_model(lstm_encoder.eval(), model_file_name)
        model_file_name = os.path.join(save_dir_models, 'lstm_decoder.pt')
        save_model(lstm_decoder.eval(), model_file_name)
        test_case="case1"
        year=2017
        # print("{}_year_{}_ID_{}".format(test_case,year,crystal_id))
        # for test_file_201X
        fig_name_cali = os.path.join(save_dir_vis_data, '{}_cali_original_ID_{}.png'.format(year,crystal_id))
        fig_name_scaled_cali = os.path.join(save_dir_vis_data, '{}_cali_scaled_ID_{}.png'.format(year,crystal_id))
        test_file_201X=df_1[df_1["xtal_id"]==crystal_id].copy()
        test_file_201X["laser_datetime"]=pd.to_datetime(test_file_201X["laser_datetime"])
        test_file_201X=test_file_201X[test_file_201X["laser_datetime"].dt.year==year].reset_index(drop=True)
        test_file_201X["delta_lumi"]=test_file_201X["int_deliv_inv_ub"].diff()
        test_file_201X.loc[0,"delta_lumi"]=0
        csv_path = os.path.join(save_dir_vis_data, f"test_{year}_ID_{crystal_id}.csv")
        test_file_201X.to_csv(csv_path, index=False)
        ecal_dataset_prep_test_201X = ECAL_Dataset_Prep(csv_path, 
                                                        input_len, 
                                                        output_len, 
                                                        stride, 
                                                        fig_name_cali, 
                                                        fig_name_scaled_cali,
                                                        verbose=False,
                                                        plt_show=False)
        ecal_dataset_prep_test_201X.start_processing()
        # check its prediction on test data-201X
        # Please note that here, the data are in the numpy format, not the tensor format
        Xtrain = ecal_dataset_prep_test_201X.np_X
        Ytrain = ecal_dataset_prep_test_201X.np_Y
        df = ecal_dataset_prep_test_201X.df_lumi
        scaler_cali = ecal_dataset_prep_test_201X.scaler_cali
        os.remove(csv_path)
        
        fig_name_mape = os.path.join(save_dir_case1_fig, '0_MAPE_{}_{}_ID_{}.png'.format(test_case,year,crystal_id))
        fig_name_mse = os.path.join(save_dir_case1_fig, '1_MSE_{}_{}_ID_{}.png'.format(test_case,year,crystal_id))
        metric_file = os.path.join(save_dir_case1_csv, '{}_{}_ID_{}.csv'.format(test_case,year,crystal_id))

        seq2seq_prediction = Seq2Seq_Prediction(lstm_encoder,
                                                lstm_decoder,
                                                Xtrain,
                                                Ytrain,
                                                df,
                                                scaler_cali,
                                                device,
                                                fig_name_mape,
                                                fig_name_mse,
                                                metric_file,
                                                test_case,
                                               plt_show=False)
        seq2seq_prediction.start_prediction()
        



