import os
import pickle

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, f1_score
from itertools import chain
import warnings
import csv

from train import warmup, train
from codivide_utils import gmm_probabilities
from utils import save_net_optimizer_to_ckpt

# Implementation
def compute_unc_weights(target, predicted, weight_mode="acc"):
    """ Uses the labels and predictions to score the results"""

    if weight_mode == "acc":
        conf_mat = confusion_matrix(target, predicted)
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        scores = conf_mat.diagonal()

    elif weight_mode == "f1_score":
        scores = f1_score(target, predicted, average=None)

    else:
        warnings.warn(f"Method {weight_mode} not implemented, using acc")
        conf_mat = confusion_matrix(target, predicted)
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        scores = conf_mat.diagonal()

    compl_scores = 1 - scores  # We use the complementary scores
    return compl_scores


def weight_smoothing(weights, num_class, lambda_w_eps, window_mode="mean"):
    """Avoids unstable behavior of weights
    weight: weight matrix of size : ( epoch x class)
    method: method to reduce the weight matrix dimension 0 (result = class_dim)
        Types:
            Simple moving average( un-weighted mean)

    TODO:   Cummulative moving average ?
            Weighted moving average
            Exponential moving average
    """

    #print("Weights to be smoothed: (weight_smoothing function)")
    #for l in np.round(weights, 4).tolist():
    #    print("\t", end="")
    #    print(l, end="\n")

    if len(weights.shape) < 2:
        weights = np.expand_dims(weights, 0)
    smooth_w = np.zeros((weights.shape[1]))

    if window_mode == "mean":
        smooth_w = np.mean(weights, axis=0)

    elif window_mode == "exp_smooth":
        alpha = 0.5
        w_dtype = weights.dtype
        scaling_factors = np.power(1. - alpha, np.arange(weights.shape[0],
                                                         dtype=w_dtype),
                                   dtype=w_dtype)
        smooth_w = np.average(weights, axis=0, weights=scaling_factors)

    else:
        warnings.warn(f"Method {window_mode} not implemented")
        smooth_w = weights[0,]

    # setting a lower bound
    smooth_w = np.maximum(lambda_w_eps, smooth_w)
    # Normalizing so that all the weights add up to num_class
    smooth_w = smooth_w / smooth_w.sum() * num_class

    print("Smoothed weights (weight_smoothing function): ")
    print("\t", end="")
    for l in np.round(smooth_w, 4).tolist():
        print(l, end=" ")
    print("\n")

    return np.round(smooth_w, 5).tolist()


def save_losses(input_loss, exp):
    name = './stats/cifar100/losses{}.pcl'
    nm = name.format(exp)
    if os.path.exists(nm):
        loss_history = pickle.load(open(nm, "rb"))
    else:
        loss_history, clean_history = [], []
    loss_history.append(input_loss)
    pickle.dump(loss_history, open(nm, "wb"))


def eval_train( model, eval_loader, CE, all_loss, epoch, net, device, r, stats_log, 
                weight_mode, log_name, codivide_policy, codivide_log, p_threshold):
    print(f'\nCo-Divide net{net}')
    model.eval()
    losses = torch.zeros(50000)
    losses_clean = torch.zeros(50000)
    targets_all = []
    targets_clean_all = []
    predictions_all = []
    with torch.no_grad():
        for batch_idx, (inputs, _, targets, index, targets_clean) in enumerate(eval_loader):
            inputs, targets, targets_clean = inputs.to(device), targets.to(device), targets_clean.to(device)
            outputs = model(inputs)
            targets_all += targets.tolist()
            targets_clean_all += targets_clean.tolist()
            _, predicted = torch.max(outputs, 1)
            predictions_all.append(predicted.tolist())
            predictions_merged = list(chain(*predictions_all))  # TD probably torch/numpy has a function for this

            loss = CE(outputs, targets)
            clean_loss = CE(outputs, targets_clean)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                losses_clean[index[b]] = clean_loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    history = torch.stack(all_loss)
    log_name.format(epoch)
    with open(log_name.format(epoch), 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)     
        # writing the fields
        #print(len(targets_all)) #50k
        #print(len(losses)) # 50k
        #print(len(all_loss)) # 2
        #print(len(predictions_merged)) # 50 k
        #print(len(predictions_all)) # 782

        csvwriter.writerow( ['Target', 'Loss', 'Prediction'] )
        for i in range(len(targets_all)):
            csvwriter.writerow( [targets_all[i], losses[i].item(), predictions_merged[i]] )
        
        #csvwriter.writerow(targets_all) 
        #csvwriter.writerow(losses) 
        #csvwriter.writerow(predictions_all)

    weights_raw = compute_unc_weights(targets_all, predictions_merged, weight_mode)
    #print("\nRaw weights: (eval_train function)")
    #print("\t", end="")
    #for l in np.round(weights_raw, 4).tolist():
    #    print(l, end=" ")
    #print("\n")

    if r >= 0.9:  # average loss over last 5 epochs to improve convergence stability
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # exp = '_std_tpc_oracle'
    # save_losses(input_loss, exp)

    prob = codivide_policy(input_loss, stats_log, epoch, net, p_threshold, np.asarray(targets_all), 
                            np.asarray(targets_clean_all), codivide_log)

    return prob, all_loss, losses_clean, weights_raw


def run_test(epoch, net1, net2, test_loader, device, test_log):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()


def run_train_loop(net1, optimizer1, sched1, net2, optimizer2, sched2, criterion, CEloss, CE, loader, p_threshold,
                   warm_up, num_epochs, all_loss, batch_size, num_class, device, lambda_u, T, alpha, noise_mode,
                   dataset, r, conf_penalty, stats_log, loss_log, test_log, weights_log, training_losses_log, log_name,
                   window_size, window_mode, lambda_w_eps, weight_mode, experiment_name, weightsLu, weightsLr, codivide_policy,
                   codivide_log, model_checkpoint_folder):
    weight_hist_1 = np.zeros((window_size, num_class))
    weight_hist_2 = np.zeros((window_size, num_class))

    for epoch in range(1, num_epochs + 1):
        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train')

        if epoch <= warm_up:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader, CEloss, conf_penalty, device, dataset, r, num_epochs,
                   noise_mode)
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader, CEloss, conf_penalty, device, dataset, r, num_epochs,
                   noise_mode)

            prob1, all_loss[0], losses_clean1, _ = eval_train(net1, eval_loader, CE, all_loss[0], epoch, 1,
                                                                         device, r, stats_log, weight_mode, log_name, 
                                                                         codivide_policy, codivide_log, p_threshold)
            prob2, all_loss[1], losses_clean2, _ = eval_train(net2, eval_loader, CE, all_loss[1], epoch, 2,
                                                                         device, r, stats_log, weight_mode, log_name, 
                                                                         codivide_policy, codivide_log, p_threshold)

            p_thr2 = np.clip(p_threshold, prob2.min() + 1e-5, prob2.max() - 1e-5)
            pred2 = prob2 > p_thr2

            loss_log.write('{},{},{},{},{}\n'.format(epoch, losses_clean2[pred2].mean(), losses_clean2[pred2].std(),
                                                     losses_clean2[~pred2].mean(), losses_clean2[~pred2].std()))
            loss_log.flush()
            loader.run('train', pred2, prob2)  # count metrics
        else:
            prob2, all_loss[1], losses_clean2, weights2_raw = eval_train(net2, eval_loader, CE, all_loss[1], epoch, 2,
                                                                         device, r, stats_log, weight_mode, log_name, 
                                                                         codivide_policy, codivide_log, p_threshold)
            prob1, all_loss[0], losses_clean1, weights1_raw = eval_train(net1, eval_loader, CE, all_loss[0], epoch, 1,
                                                                         device, r, stats_log, weight_mode, log_name, 
                                                                         codivide_policy, codivide_log, p_threshold)

            # Updating weight history
            weight_hist_1[1:] = weight_hist_1[:-1]
            weight_hist_1[0, :] = weights1_raw
            weight_hist_2[1:] = weight_hist_2[:-1]
            weight_hist_2[0, :] = weights2_raw

            if epoch < (warm_up + window_size):
                weights1_smooth = weight_smoothing(weight_hist_1[:epoch - warm_up], num_class, lambda_w_eps,
                                                   window_mode=window_mode)
                weights2_smooth = weight_smoothing(weight_hist_2[:epoch - warm_up], num_class, lambda_w_eps,
                                                   window_mode=window_mode)
            else:
                weights1_smooth = weight_smoothing(weight_hist_1, num_class, lambda_w_eps,
                                                   window_mode=window_mode)
                weights2_smooth = weight_smoothing(weight_hist_2, num_class, lambda_w_eps,
                                                   window_mode=window_mode)
            # Write the weights to file
            # creating a csv writer object 
            csvwriter = csv.writer(weights_log)
            #print([epoch])
            #print(weights1_raw)
            #print(weights2_raw)
            #print("(1) The type of weights1_raw is : ", type(weights1_raw))
            #print("(1) The size of weights1_raw is : ", len(weights1_raw))

            #print("(2) The type of weights1_smooth is : ", type(weights1_smooth))
            #print("(2) The size of weights1_smooth is : ", len(weights1_smooth))

            csvwriter.writerow([epoch] + weights1_raw.tolist() + weights2_raw.tolist())
            csvwriter.writerow([epoch] + weights1_smooth + weights2_smooth)
            weights_log.flush()

            print('\nTrain Net1')
            p_thr2 = np.clip(p_threshold, prob2.min() + 1e-5, prob2.max() - 1e-5)
            pred2 = prob2 > p_thr2

            loss_log.write('{},{},{},{},{}\n'.format(epoch, losses_clean2[pred2].mean(), losses_clean2[pred2].std(),
                                                     losses_clean2[~pred2].mean(), losses_clean2[~pred2].std()))
            loss_log.flush()

            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
            train(epoch, net1, net2, criterion, optimizer1, labeled_trainloader, unlabeled_trainloader, lambda_u,
                  batch_size, num_class, device, T, alpha, warm_up, dataset, r, noise_mode, num_epochs,
                  weights1_smooth, training_losses_log, weightsLu, weightsLr)  # train net1

            print('\nTrain Net2')
            # prob1, all_loss[0], losses_clean1, weights1_raw = eval_train(net1, eval_loader, CE, all_loss[0], epoch, 1,
            #                                               device, r, stats_log)

            p_thr1 = np.clip(p_threshold, prob1.min() + 1e-5, prob1.max() - 1e-5)
            pred1 = prob1 > p_thr1

            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
            train(epoch, net2, net1, criterion, optimizer2, labeled_trainloader, unlabeled_trainloader, lambda_u,
                  batch_size, num_class, device, T, alpha, warm_up, dataset, r, noise_mode, num_epochs,
                  weights2_smooth, training_losses_log, weightsLu, weightsLr)  # train net2

            if model_checkpoint_folder and (not epoch%5 or epoch ==9):
                print(f'[ SAVING MODELS] EPOCH: {epoch} PATH: {model_checkpoint_folder}')
                save_net_optimizer_to_ckpt(net1, optimizer1, f'{model_checkpoint_folder}/{epoch}_1.pt')
                save_net_optimizer_to_ckpt(net2, optimizer2, f'{model_checkpoint_folder}/{epoch}_2.pt')

        run_test(epoch, net1, net2, test_loader, device, test_log)

        sched1.step()
        sched2.step()
    final_checkpoint_name = './final_checkpoints/%s_final_checkpoint.pth.tar' % experiment_name
    torch.save(net1.state_dict(), final_checkpoint_name)
