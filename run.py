# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import shutil
import argparse
import datetime
import pickle
import time

import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import sys
sys.path.append('../')

import mxnet as mx
mx.random.seed(int(time.time()))

from utils import gpu_helper, gpu_exists, calc_topic_uniqueness, get_topic_words_decoder_weights, request_pmi, print_topic_with_scores, print_topics


from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Training WAE in MXNet')
    parser.add_argument('-dom','--domain', type=str, default='twenty_news', help='domain to run', required=False)
    parser.add_argument('-data','--data_path', type=str, default='', help='file path for dataset', required=False)
    parser.add_argument('-max_labels','--max_labels', type=int, default=100, help='max number of topics to specify as labels for a single training document', required=False)
    parser.add_argument('-max_labeled_samples','--max_labeled_samples', type=int, default=10, help='max number of labeled samples per topic', required=False)
    parser.add_argument('-label_seed','--label_seed', type=lambda x: int(x) if x != 'None' else None, default=None, help='random seed for subsampling the labeled dataset', required=False)
    parser.add_argument('-mod','--model', type=str, default='dirichlet', help='model to use', required=False)
    parser.add_argument('-desc','--description', type=str, default='', help='description for the experiment', required=False)
    parser.add_argument('-alg','--algorithm', type=str, default='standard', help='algorithm to use for training: standard', required=False)
    parser.add_argument('-bs','--batch_size', type=int, default=256, help='batch_size for training', required=False)
    parser.add_argument('-opt','--optim', type=str, default='Adam', help='encoder training algorithm', required=False)
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-4, help='learning rate', required=False)
    parser.add_argument('-l2','--weight_decay', type=float, default=0., help='weight decay', required=False)
    parser.add_argument('-e_nh','--enc_n_hidden', type=int, nargs='+', default=[128], help='# of hidden units for encoder or list of hiddens for each layer', required=False)
    parser.add_argument('-e_nl','--enc_n_layer', type=int, default=1, help='# of hidden layers for encoder, set to -1 if passing list of n_hiddens', required=False)
    parser.add_argument('-e_nonlin','--enc_nonlinearity', type=str, default='sigmoid', help='type of nonlinearity for encoder', required=False)
    parser.add_argument('-e_weights','--enc_weights', type=str, default='', help='file path for encoder weights', required=False)
    parser.add_argument('-e_freeze','--enc_freeze', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to freeze the encoder weights', required=False)
    parser.add_argument('-lat_nonlin','--latent_nonlinearity', type=str, default='', help='type of to use prior to decoder', required=False)
    parser.add_argument('-d_nh','--dec_n_hidden', type=int, nargs='+', default=[128], help='# of hidden units for decoder or list of hiddens for each layer', required=False)
    parser.add_argument('-d_nl','--dec_n_layer', type=int, default=0, help='# of hidden layers for decoder', required=False)
    parser.add_argument('-d_nonlin','--dec_nonlinearity', type=str, default='', help='type of nonlinearity for decoder', required=False)
    parser.add_argument('-d_weights','--dec_weights', type=str, default='', help='file path for decoder weights', required=False)
    parser.add_argument('-d_freeze','--dec_freeze', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to freeze the decoder weights', required=False)
    parser.add_argument('-d_word_dist','--dec_word_dist', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to init decoder weights with training set word distributions', required=False)
    parser.add_argument('-dis_nh','--dis_n_hidden', type=int, nargs='+', default=[128], help='# of hidden units for encoder or list of hiddens for each layer', required=False)
    parser.add_argument('-dis_nl','--dis_n_layer', type=int, default=1, help='# of hidden layers for encoder, set to -1 if passing list of n_hiddens', required=False)
    parser.add_argument('-dis_nonlin','--dis_nonlinearity', type=str, default='sigmoid', help='type of nonlinearity for discriminator', required=False)
    parser.add_argument('-dis_y_weights','--dis_y_weights', type=str, default='', help='file path for discriminator_y weights', required=False)
    parser.add_argument('-dis_z_weights','--dis_z_weights', type=str, default='', help='file path for discriminator_z weights', required=False)
    parser.add_argument('-dis_freeze','--dis_freeze', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to freeze the encoder weights', required=False)
    parser.add_argument('-include_w','--include_weights', type=str, nargs='*', default=[], help='weights to train on (default is all weights) -- all others are kept fixed; Ex: E.z_encoder D.decoder', required=False)
    parser.add_argument('-eps','--epsilon', type=float, default=1e-8, help='epsilon param for Adam', required=False)
    parser.add_argument('-mx_it','--max_iter', type=int, default=50001, help='max # of training iterations', required=False)
    parser.add_argument('-train_stats_every','--train_stats_every', type=int, default=100, help='skip train_stats_every iterations between recording training stats', required=False)
    parser.add_argument('-eval_stats_every','--eval_stats_every', type=int, default=100, help='skip eval_stats_every iterations between recording evaluation stats', required=False)
    parser.add_argument('-ndim_y','--ndim_y', type=int, default=256, help='dimensionality of y - topic indicator', required=False)
    parser.add_argument('-ndim_x','--ndim_x', type=int, default=2, help='dimensionality of p(x) - data distribution', required=False)
    parser.add_argument('-saveto','--saveto', type=str, default='', help='path prefix for saving results', required=False)
    parser.add_argument('-gpu','--gpu', type=int, default=-2, help='if/which gpu to use (-1: all, -2: None)', required=False)
    parser.add_argument('-hybrid','--hybridize', type=lambda x: (str(x).lower() == 'true'), default=False, help='declaritive True (hybridize) or imperative False', required=False)
    parser.add_argument('-full_npmi','--full_npmi', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to compute NPMI for full trajectory', required=False)
    parser.add_argument('-eot','--eval_on_test', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to evaluate on the test set (True) or validation set (False)', required=False)
    parser.add_argument('-verb','--verbose', type=lambda x: (str(x).lower() == 'true'), default=True, help='whether to print progress to stdout', required=False)
    parser.add_argument('-dirich_alpha','--dirich_alpha', type=float, default=1e-1, help='param for Dirichlet prior', required=False)
    parser.add_argument('-adverse','--adverse', type=lambda x: (str(x).lower() == 'true'), default=True, help='whether to turn on adverserial training (MMD or GAN). set to False if only train auto-encoder', required=False)
    parser.add_argument('-update_enc','--update_enc', type=lambda x: (str(x).lower() == 'true'), default=True, help='whether to update encoder for unlabed_train_op()', required=False)
    parser.add_argument('-labeled_loss_lambda','--labeled_loss_lambda', type=float, default=1.0, help='param for Dirichlet noise for label', required=False)
    parser.add_argument('-train_mode','--train_mode', type=str, default='mmd', help="set to mmd or adv (for GAN)", required=False)
    parser.add_argument('-kernel_alpha','--kernel_alpha', type=float, default=1.0, help='param for information diffusion kernel', required=False)
    parser.add_argument('-recon_alpha','--recon_alpha', type=float, default=-1.0, help='multiplier of the reconstruction loss when combined with mmd loss', required=False)
    parser.add_argument('-recon_alpha_adapt','--recon_alpha_adapt', type=float, default=-1.0, help='adaptively change recon_alpha so that [total loss = mmd + recon_alpha_adapt * recon loss], set to -1 if no adapt', required=False)
    parser.add_argument('-dropout_p','--dropout_p', type=float, default=-1.0, help='dropout probability in encoder', required=False)
    parser.add_argument('-l2_alpha','--l2_alpha', type=float, default=-1.0, help='alpha multipler for L2 regularization on latent vector', required=False)
    parser.add_argument('-latent_noise','--latent_noise', type=float, default=0.0, help='proportion of dirichlet noise added to the latent vector after softmax', required=False)
    parser.add_argument('-topic_decoder_weight','--topic_decoder_weight', type=lambda x: (str(x).lower() == 'true'), default=False, help='extract topic words based on decoder weights or decoder outputs', required=False)
    parser.add_argument('-retrain_enc_only','--retrain_enc_only', type=lambda x: (str(x).lower() == 'true'), default=False, help='only retrain the encoder for reconstruction loss', required=False)
    parser.add_argument('-l2_alpha_retrain','--l2_alpha_retrain', type=float, default=0.1, help='alpha multipler for L2 regularization on encoder output during retraining', required=False)
    args = vars(parser.parse_args())


    if args['domain'] == 'twenty_news_sklearn':
        from examples.domains.twenty_news_sklearn_wae import TwentyNews as Domain
    elif args['domain'] == 'wikitext-103':
        from examples.domains.wikitext103_wae import Wikitext103 as Domain
    elif args['domain'] == 'nytimes-pbr':
        from examples.domains.nyt_wae import Nytimes as Domain
    elif args['domain'] == 'ag_news_csv':
        from examples.domains.ag_news_wae import Agnews as Domain
    elif args['domain'] == 'dbpedia_csv':
        from examples.domains.dbpedia_wae import Dbpedia as Domain
    elif args['domain'] == 'yelp_review_polarity_csv':
        from examples.domains.yelp_polarity_wae import YelpPolarity as Domain
    elif args['domain'] == 'lda_synthetic':
        from examples.domains.lda_synthetic import LdaSynthetic as Domain
    else:
        raise NotImplementedError(args['domain'])

    if args['model'] == 'dirichlet':
        from models.dirichlet import Encoder, Decoder, Discriminator_y
    else:
        raise NotImplementedError(args['model'])

    from compute_op import Unsupervised as Compute

    assert args['latent_noise'] >= 0 and args['latent_noise'] <= 1
    if args['description'] == '':
        args['description'] = args['domain'] + '-' + args['algorithm'] + '-' + args['model']
        if args['un_label_coeffs'][0] > 0 and args['un_label_coeffs'][1] == 0:
            args['description'] += '-unsup'
        elif args['un_label_coeffs'][0] > 0 and args['un_label_coeffs'][1] > 0:
            args['description'] += '-semisup'
        else:
            args['description'] += '-sup'
    elif args['description'].isdigit():
        args['description'] = args['domain'] + '-' + args['algorithm'] + '-' + args['model'] + '-' + args['description']

    if args['saveto'] == '':
        args['saveto'] = 'examples/results/' + args['description'].replace('-','/')

    saveto = args['saveto'] + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S/{}').format('')
    if not os.path.exists(saveto):
        os.makedirs(saveto)
        os.makedirs(saveto + '/weights/encoder')
        os.makedirs(saveto + '/weights/decoder')
        os.makedirs(saveto + '/weights/discriminator_y')
        os.makedirs(saveto + '/weights/discriminator_z')
    shutil.copy(os.path.realpath('compute_op.py'), os.path.join(saveto, 'compute_op.py'))
    shutil.copy(os.path.realpath('core.py'), os.path.join(saveto, 'core.py'))
    shutil.copy(os.path.realpath('run.py'), os.path.join(saveto, 'run.py'))
    shutil.copy(os.path.realpath('utils.py'), os.path.join(saveto, 'utils.py'))

    # domain_file = args['domain']+'.py'
    # shutil.copy(os.path.realpath('examples/domains/'+domain_file), os.path.join(saveto, domain_file))
    model_file = args['model']+'.py'
    shutil.copy(os.path.realpath('models/'+model_file), os.path.join(saveto, model_file))
    args['saveto'] = saveto
    with open(saveto+'args.txt', 'w') as file:
        for key, val in args.items():
            if val != '':
                if isinstance(val, list) or isinstance(val, tuple):
                    val = [str(v) for v in val]
                    file.write('--'+str(key)+' '+' '.join(val)+'\n')
                else:
                    file.write('--'+str(key)+' '+str(val)+'\n')

    if args['gpu'] >= 0 and gpu_exists(args['gpu']):
        args['description'] += ' (gpu'+str(args['gpu'])+')'
    else:
        args['description'] += ' (cpu)'

    pickle.dump(args, open(args['saveto']+'args.p','wb'))

    return Compute, Domain, Encoder, Decoder, Discriminator_y, args


def run_experiment(Compute, Domain, Encoder, Decoder, Discriminator_y, args):
    print('\nSaving to: '+args['saveto'])

    model_ctx = gpu_helper(args['gpu'])

    data = Domain(batch_size=args['batch_size'], data_path=args['data_path'], ctx=model_ctx, saveto=args['saveto'])
    print('train dimension = ', data.data['train'].shape)
    if type(data.data['train']) is np.ndarray:
        mean_length = np.mean(np.sum(data.data['train'], axis=1))
    else:
        mean_length = mx.nd.mean(mx.nd.sum(data.data['train'], axis=1)).asscalar()
    vocab_size = data.data['train'].shape[1]
    if data.data['train_with_labels'] is not None:
        print('train_with_labels dimension = ', data.data['train_with_labels'].shape)

    if args['recon_alpha'] < 0:
        args['recon_alpha'] = 1.0 / (mean_length * np.log(vocab_size))
    print('Setting recon_alpha to {}'.format(args['recon_alpha']))

    Enc = Encoder(model_ctx=model_ctx, batch_size=args['batch_size'], input_dim=args['ndim_x'], ndim_y=args['ndim_y'],
                  n_hidden=args['enc_n_hidden'], n_layers=args['enc_n_layer'], nonlin=args['enc_nonlinearity'],
                  weights_file=args['enc_weights'], freeze=args['enc_freeze'], latent_nonlin=args['latent_nonlinearity'])
    Dec = Decoder(model_ctx=model_ctx, batch_size=args['batch_size'], output_dim=args['ndim_x'], ndim_y=args['ndim_y'],
                  n_hidden=args['dec_n_hidden'], n_layers=args['dec_n_layer'], nonlin=args['dec_nonlinearity'],
                  weights_file=args['dec_weights'], freeze=args['dec_freeze'], latent_nonlin=args['latent_nonlinearity'])
    Dis_y = Discriminator_y(model_ctx=model_ctx, batch_size=args['batch_size'], ndim_y=args['ndim_y'],
                            n_hidden=args['dis_n_hidden'], n_layers=args['dis_n_layer'],
                            nonlin=args['dis_nonlinearity'], weights_file=args['dis_y_weights'],
                            freeze=args['dis_freeze'], latent_nonlin=args['latent_nonlinearity'])
    if args['enc_weights']:
        Enc.load_parameters(args['enc_weights'], ctx=model_ctx)
    else:
        Enc.init_weights()
    if args['dec_weights']:
        Dec.load_parameters(args['dec_weights'], ctx=model_ctx)
    else:
        Dec.init_weights()
    Dis_y.init_weights()
    # load pre-trained document classifier
    if args['hybridize']:
        print('NOTE: Hybridizing Encoder and Decoder (Declaritive mode).')
        Enc.hybridize()
        Dec.hybridize()
        Dis_y.hybridize()
    else:
        print('NOTE: Not Hybridizing Encoder and Decoder (Imperative mode).')

    compute = Compute(data, Enc, Dec,  Dis_y, args)

    N_train = data.data['train'].shape[0]

    epochs = range(args['max_iter'])
    if args['verbose']:
        print(' ')
        epochs = tqdm(epochs, desc=args['description'])

    train_record = {'loss_discriminator':[], 'loss_generator':[], 'loss_reconstruction':[], 'latent_max_distr':[],
                    'latent_avg_entropy':[], 'latent_avg':[], 'dirich_avg_entropy':[], 'loss_labeled':[]}
    eval_record = {'NPMI':[], 'Topic Uniqueness':[], 'Top Words':[],
                   'NPMI2':[], 'Topic Uniqueness2':[], 'Top Words2':[],
                   'u_loss_train':[], 'l_loss_train':[],
                   'u_loss_val':[], 'l_loss_val':[],
                   'u_loss_test':[], 'l_loss_test':[],
                   'l_acc_train':[], 'l_acc_val':[], 'l_acc_test':[]}

    total_iterations_train = N_train // args['batch_size']
    training_start_time = time.time()
    i = 0
    if args['retrain_enc_only']:
        print('Retraining encoder ONLY!')
        for i in epochs:
            sum_loss_autoencoder = 0.0
            epoch_start_time = time.time()
            for itr in range(total_iterations_train):
                loss_reconstruction = compute.retrain_enc(args['l2_alpha_retrain'])
                sum_loss_autoencoder += loss_reconstruction
            if args['verbose']:
                # epochs.set_postfix({'L_Dis': loss_discriminator, 'L_Gen': loss_generator, 'L_Recon': loss_reconstruction})
                print("Epoch {} done in {} sec - loss: a={:.5g} - total {} min".format(
                    i + 1, int(time.time() - epoch_start_time),
                    sum_loss_autoencoder / total_iterations_train,
                    int((time.time() - training_start_time) // 60)))
    else:
        for i in epochs:
            sum_loss_generator = 0.0
            sum_loss_discriminator = 0.0
            sum_loss_autoencoder = 0.0
            sum_discriminator_z_confidence_true = 0.0
            sum_discriminator_z_confidence_fake = 0.0
            sum_discriminator_y_confidence_true = 0.0
            sum_discriminator_y_confidence_fake = 0.0
            sum_loss_labeled = 0.0

            latent_max_distr = np.zeros(args['ndim_y'])
            latent_entropy_avg = 0.0
            latent_v_avg = np.zeros(args['ndim_y'])
            dirich_avg_entropy = 0.0

            epoch_start_time = time.time()
            for itr in range(total_iterations_train):
                if args['train_mode'] == 'mmd':
                    loss_reconstruction, loss_discriminator, latent_max, latent_entropy, latent_v, dirich_entropy = \
                        compute.unlabeled_train_op_mmd_combine(update_enc=args['update_enc'])
                    loss_generator, \
                    discriminator_z_confidence_true, discriminator_z_confidence_fake, \
                    discriminator_y_confidence_true, discriminator_y_confidence_fake = 0,0,0,0,0
                elif args['train_mode'] == 'adv':
                    loss_discriminator, loss_generator, loss_reconstruction, \
                    discriminator_z_confidence_true, discriminator_z_confidence_fake, \
                    discriminator_y_confidence_true, discriminator_y_confidence_fake, \
                    latent_max, latent_entropy, latent_v, dirich_entropy = \
                        compute.unlabeled_train_op_adv_combine_add(update_enc=args['update_enc'])

                sum_loss_discriminator += loss_discriminator
                sum_loss_generator += loss_generator
                sum_loss_autoencoder += loss_reconstruction
                sum_discriminator_z_confidence_true += discriminator_z_confidence_true
                sum_discriminator_z_confidence_fake += discriminator_z_confidence_fake
                sum_discriminator_y_confidence_true += discriminator_y_confidence_true
                sum_discriminator_y_confidence_fake += discriminator_y_confidence_fake

                latent_max_distr += latent_max
                latent_entropy_avg += latent_entropy
                latent_v_avg += latent_v
                dirich_avg_entropy += dirich_entropy

            train_record['loss_discriminator'].append(sum_loss_discriminator / total_iterations_train)
            train_record['loss_generator'].append(sum_loss_generator / total_iterations_train)
            train_record['loss_reconstruction'].append(sum_loss_autoencoder / total_iterations_train)
            train_record['latent_max_distr'].append(latent_max_distr / total_iterations_train)
            train_record['latent_avg_entropy'].append(latent_entropy_avg / total_iterations_train)
            train_record['latent_avg'].append(latent_v_avg / total_iterations_train)
            train_record['dirich_avg_entropy'].append(dirich_avg_entropy / total_iterations_train)
            train_record['loss_labeled'].append(sum_loss_labeled / total_iterations_train)
            if args['verbose']:
                # epochs.set_postfix({'L_Dis': loss_discriminator, 'L_Gen': loss_generator, 'L_Recon': loss_reconstruction})
                print("Epoch {} done in {} sec - loss: g={:.5g}, d={:.5g}, a={:.5g}, label={:.5g} - disc_z: true={:.1f}%, fake={:.1f}% - disc_y: true={:.1f}%, fake={:.1f}% - total {} min".format(
                    i + 1, int(time.time() - epoch_start_time),
                    sum_loss_generator / total_iterations_train,
                    sum_loss_discriminator / total_iterations_train,
                    sum_loss_autoencoder / total_iterations_train,
                    sum_loss_labeled / total_iterations_train,
                    sum_discriminator_z_confidence_true / total_iterations_train * 100,
                    sum_discriminator_z_confidence_fake / total_iterations_train * 100,
                    sum_discriminator_y_confidence_true / total_iterations_train * 100,
                    sum_discriminator_y_confidence_fake / total_iterations_train * 100,
                    int((time.time() - training_start_time) // 60)))
                print('Latent avg entropy = {}, dirich_entropy={}'.format(
                    train_record['latent_avg_entropy'][-1], train_record['dirich_avg_entropy'][-1]))
                # print(train_record['latent_avg'][-1])
            if i == (args['max_iter'] - 1) or (args['eval_stats_every'] > 0 and i % args['eval_stats_every'] == 0):
                if args['recon_alpha_adapt'] > 0 and i == 0:
                    compute.args['recon_alpha'] = train_record['loss_discriminator'][-1] / \
                                                  train_record['loss_reconstruction'][-1]
                    compute.args['recon_alpha'] = abs(compute.args['recon_alpha']) * args['recon_alpha_adapt']
                    print("recon_alpha adjusted to {}".format(compute.args['recon_alpha']))

                if args['domain'] == 'synthetic':
                    enc_out = compute.test_synthetic_op()
                    np.save(os.path.join(args['saveto'], "enc_out_epoch{}".format(i)), enc_out.asnumpy())
                else:
                    # extract topic words from decoder output:
                    topic_words = get_topic_words_decoder_weights(Dec, data, model_ctx, decoder_weights=False)
                    topic_uniqs = calc_topic_uniqueness(topic_words)
                    eval_record['Topic Uniqueness'].append(np.mean(list(topic_uniqs.values())))
                    topic_json = dict()
                    for tp in range(len(topic_words)):
                        topic_json[tp] = topic_words[tp]
                    pmi_dict, npmi_dict = request_pmi(topic_dict=topic_json, port=1234)
                    eval_record['NPMI'].append(np.mean(list(npmi_dict.values())))
                    print("Topic Eval (decoder output): Uniq={:.5g}, NPMI={:.5g}".format(
                        eval_record['Topic Uniqueness'][-1], eval_record['NPMI'][-1]))
                    eval_record['Top Words'].append(topic_json)
                    print_topics(topic_json, npmi_dict, topic_uniqs, data)

                    # extract topic words from decoder weight matrix:
                    topic_words = get_topic_words_decoder_weights(Dec, data, model_ctx, decoder_weights=True)
                    topic_uniqs = calc_topic_uniqueness(topic_words)
                    eval_record['Topic Uniqueness2'].append(np.mean(list(topic_uniqs.values())))
                    topic_json = dict()
                    for tp in range(len(topic_words)):
                        topic_json[tp] = topic_words[tp]
                    pmi_dict, npmi_dict = request_pmi(topic_dict=topic_json, port=1234)
                    eval_record['NPMI2'].append(np.mean(list(npmi_dict.values())))
                    print("Topic Eval (decoder weight): Uniq={:.5g}, NPMI={:.5g}".format(
                        eval_record['Topic Uniqueness2'][-1], eval_record['NPMI2'][-1]))
                    eval_record['Top Words2'].append(topic_json)
                    print_topics(topic_json, npmi_dict, topic_uniqs, data)

                    # evaluate train, validate and test losses for w/ w/o labels:
                    u_loss_train, l_loss_train, u_loss_val, l_loss_val, u_loss_test, l_loss_test, l_acc_train, \
                    l_acc_val, l_acc_test = 0,0,0,0,0,0,0,0,0
                    u_loss_train, l_loss_train, l_acc_train = compute.test_op(dataset='train')
                    eval_record['u_loss_train'].append(u_loss_train)
                    eval_record['l_loss_train'].append(l_loss_train)
                    eval_record['l_acc_train'].append(l_acc_train)
                    if data.data['valid'] is not None:
                        u_loss_val, l_loss_val, l_acc_val = compute.test_op(dataset='valid')
                        eval_record['u_loss_val'].append(u_loss_val)
                        eval_record['l_loss_val'].append(l_loss_val)
                        eval_record['l_acc_val'].append(l_acc_val)
                    if data.data['test'] is not None:
                        u_loss_test, l_loss_test, l_acc_test = compute.test_op(dataset='test')
                        eval_record['u_loss_test'].append(u_loss_test)
                        eval_record['l_loss_test'].append(l_loss_test)
                        eval_record['l_acc_test'].append(l_acc_test)
                    print("Train loss u-l-acc: {:.5g}-{:.5g}-{:.5g}, Val: {:.5g}-{:.5g}-{:.5g}, Test: {:.5g}-{:.5g}-{:.5g}".format(
                        u_loss_train, l_loss_train, l_acc_train, u_loss_val, l_loss_val, l_acc_val, u_loss_test, l_loss_test, l_acc_test))

                    pickle.dump(train_record,open(args['saveto']+'train_record.p','wb'))
                    pickle.dump(eval_record,open(args['saveto']+'eval_record.p','wb'))

    # save final weights
    Enc.save_parameters(args['saveto']+'weights/encoder/Enc_'+str(i))
    Dec.save_parameters(args['saveto']+'weights/decoder/Dec_'+str(i))
    Dis_y.save_parameters(args['saveto']+'weights/discriminator_y/Dis_y_'+str(i))

    # save the latent features
    compute.save_latent(args['saveto'])

    if args['domain'] == 'lda_synthetic':
        # save the decoder weight matrix
        params = Dec.collect_params()
        params = params['decoder0_dense0_weight'].data().transpose()
        np.save(args['saveto']+'decoder_weight.npy', params.asnumpy())

    # print_topic_with_scores(eval_record['Top Words'][-1])
    print('Done! ' + args['description'])
    print('\nSaved to: '+args['saveto'])

if __name__ == '__main__':
    Compute, Domain, Encoder, Decoder, Discriminator_y, args = parse_args()
    run_experiment(Compute, Domain, Encoder, Decoder, Discriminator_y, args)
