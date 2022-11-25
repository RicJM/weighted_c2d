import os
import pickle
import sys

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, f1_score
from itertools import chain
import warnings
import csv


from train_weigthed import warmup, train
from codivide_utils import per_sample_plot, enable_bn
from utils import save_net_optimizer_to_ckpt

# Implementation
def compute_unc_weights(target, predicted, weight_mode="acc"):
    """Uses the labels and predictions to score the results"""

    if weight_mode == "acc":
        conf_mat = confusion_matrix(target, predicted)
        conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
        scores = conf_mat.diagonal()

    elif weight_mode == "f1_score":
        scores = f1_score(target, predicted, average=None)

    else:
        warnings.warn(f"Method {weight_mode} not implemented, using acc")
        conf_mat = confusion_matrix(target, predicted)
        conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
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

    # print("Weights to be smoothed: (weight_smoothing function)")
    # for l in np.round(weights, 4).tolist():
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
        scaling_factors = np.power(
            1.0 - alpha, np.arange(weights.shape[0], dtype=w_dtype), dtype=w_dtype
        )
        smooth_w = np.average(weights, axis=0, weights=scaling_factors)

    else:
        warnings.warn(f"Method {window_mode} not implemented")
        smooth_w = weights[
            0,
        ]

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


def eval_train(
    model,
    eval_loader,
    CE,
    all_loss,
    epoch,
    net,
    device,
    r,
    stats_log,
    weight_mode,
    log_name,
    codivide_policy,
    codivide_log,
    p_threshold,
    num_class,
    figures_folder,
    enableLog=False,
    compute_entropy=False,
    mcbn_forward_passes=3,
    per_class_testing_accuracy=None,
):

    print(f"\nCo-Divide net{net}")
    model.eval()
    epsilon = sys.float_info.min

    forward_passes = 1
    if compute_entropy:
        forward_passes = mcbn_forward_passes

    forward_passes = 1  # TODO change this

    losses = torch.zeros(50000)
    losses_clean = torch.zeros(50000)
    softmaxs = torch.zeros(size=(50000, num_class, forward_passes), device=device)

    targets_all = torch.zeros(50000, device=device)
    targets_clean_all = torch.zeros(50000, device=device)
    predictions_all = torch.zeros(50000, device=device)

    with torch.no_grad():
        for i in range(0, forward_passes):
            if i == 1:  # to leverage BN's stochastic behaviour
                enable_bn(model)
            for batch_idx, (inputs, _, targets, index, targets_clean) in enumerate(
                eval_loader
            ):
                inputs, targets, targets_clean = (
                    inputs.to(device),
                    targets.to(device),
                    targets_clean.to(device),
                )
                outputs = model(inputs)
                softmax = torch.softmax(outputs, dim=1)  # shape (n_samples, n_classes)
                for b in range(inputs.size(0)):
                    softmaxs[index[b], :, i] = softmax[
                        b
                    ]  # shape (n_samples, n_classes, n_mcdo_passes)
                if i == 0:
                    _, predicted = torch.max(outputs, 1)
                    for j, b in enumerate(range(index.size(0))):
                        targets_all[index[b]] = targets[j]
                        targets_clean_all[index[b]] = targets_clean[j]
                        predictions_all[index[b]] = predicted[j]

                    loss = CE(outputs, targets)
                    clean_loss = CE(outputs, targets_clean)
                    for b in range(inputs.size(0)):
                        losses[index[b]] = loss[b]
                        losses_clean[index[b]] = clean_loss[b]

    per_class_training_accuracy = [
        sum(predictions_all[targets_clean_all == c] == c).item() * num_class / 50000
        for c in set(targets_clean_all.cpu().numpy().astype("int").tolist())
    ]

    predictions_merged = predictions_all.tolist()
    targets_all = targets_all.cpu().numpy().astype("int")
    targets_clean_all = targets_clean_all.cpu().numpy().astype("int")

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    # Per sample uncertainty.
    sample_mean_over_mcbn = torch.mean(softmaxs, dim=2)  # shape (n_samples, n_classes)
    sample_entropy = (
        -torch.sum(
            sample_mean_over_mcbn * torch.log(sample_mean_over_mcbn + epsilon), axis=-1
        )
        .cpu()
        .numpy()
    )  # shape (n_samples,)
    sample_entropy = (sample_entropy - sample_entropy.min()) / (
        sample_entropy.max() - sample_entropy.min()
    )

    history = torch.stack(all_loss)
    if enableLog:
        log_name = log_name.format(f"net_{net}_epoch_{epoch}")
        with open(log_name, "w", encoding="utf-8") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(["Target", "Loss", "Prediction", "Entropy"])
            for i in range(len(targets_all)):
                csvwriter.writerow(
                    [
                        targets_all.tolist()[i],
                        losses[i].item(),
                        predictions_merged[i],
                        sample_entropy[i].item(),
                    ]
                )

    weights_raw = compute_unc_weights(
        targets_all.tolist(), predictions_merged, weight_mode
    )

    if r >= 0.9:  # average loss over last 5 epochs to improve convergence stability
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    prob, gmm, ccgmm = codivide_policy(
        input_loss,
        stats_log,
        epoch,
        net,
        p_threshold,
        targets_all,
        targets_clean_all,
        codivide_log,
    )
    if enableLog:
        per_sample_plot(
            input_loss.cpu().numpy().ravel(),
            np.asarray(targets_all),
            np.asarray(targets_clean_all),
            per_class_testing_accuracy,
            per_class_training_accuracy,
            sample_entropy,
            gmm,
            ccgmm,
            p_threshold,
            figures_folder,
            epoch,
        )

    return prob, all_loss, losses_clean, weights_raw


def run_test(epoch, net1, net2, test_loader, device, test_log, num_class):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    per_class_accuracy = np.zeros(num_class)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            for c in set(predicted.cpu().numpy()):
                per_class_accuracy[c] += sum(predicted[targets == c] == c)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.0 * correct / total
    per_class_accuracy /= total / num_class
    print(f"[TEST ]: PER CLASS ACCURACY {per_class_accuracy}")
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write("Epoch:%d   Accuracy:%.2f\n" % (epoch, acc))
    test_log.flush()

    return per_class_accuracy


def run_train_loop(
    net1,
    optimizer1,
    sched1,
    net2,
    optimizer2,
    sched2,
    criterion,
    CEloss,
    CE,
    loader,
    p_threshold,
    warm_up,
    num_epochs,
    all_loss,
    batch_size,
    num_class,
    device,
    lambda_u,
    T,
    alpha,
    noise_mode,
    dataset,
    r,
    conf_penalty,
    stats_log,
    loss_log,
    test_log,
    weights_log,
    training_losses_log,
    log_name,
    window_size,
    window_mode,
    lambda_w_eps,
    weight_mode,
    experiment_name,
    weightsLu,
    weightsLr,
    enableLog,
    figures_folder,
    codivide_policy,
    codivide_log,
    model_checkpoint_folder,
    resume_epoch,
):
    weight_hist_1 = np.zeros((window_size, num_class))
    weight_hist_2 = np.zeros((window_size, num_class))
    per_class_accuracy = np.ones(num_class)

    for epoch in range(resume_epoch, num_epochs + 1):
        test_loader = loader.run("test")
        eval_loader = loader.run("BN_eval_train")  # shuffling needed to perform MCBN
        if epoch <= 10:
            p_threshold = 0.7
        else:
            p_threshold = 0.01

        if epoch <= warm_up:
            warmup_trainloader = loader.run("warmup")
            print("Warmup Net1")
            warmup(
                epoch,
                net1,
                optimizer1,
                warmup_trainloader,
                CEloss,
                conf_penalty,
                device,
                dataset,
                r,
                num_epochs,
                noise_mode,
            )
            print("\nWarmup Net2")
            warmup(
                epoch,
                net2,
                optimizer2,
                warmup_trainloader,
                CEloss,
                conf_penalty,
                device,
                dataset,
                r,
                num_epochs,
                noise_mode,
            )

            # prob1, all_loss[0], losses_clean1, _ = eval_train(net1, eval_loader, CE, all_loss[0], epoch, 1,
            #                                                   device, r, stats_log, weight_mode, log_name,
            #                                                   codivide_policy, codivide_log, p_threshold,
            #                                                   figures_folder, enableLog)

            prob2, all_loss[1], losses_clean2, _ = eval_train(
                net2,
                eval_loader,
                CE,
                all_loss[1],
                epoch,
                2,
                device,
                r,
                stats_log,
                weight_mode,
                log_name,
                codivide_policy,
                codivide_log,
                p_threshold,
                num_class,
                figures_folder,
                enableLog,
                compute_entropy=True,
                mcbn_forward_passes=3,
                per_class_testing_accuracy=per_class_accuracy,
            )

            p_thr2 = np.clip(p_threshold, prob2.min() + 1e-5, prob2.max() - 1e-5)
            pred2 = prob2 > p_thr2

            loss_log.write(
                "{},{},{},{},{}\n".format(
                    epoch,
                    losses_clean2[pred2].mean(),
                    losses_clean2[pred2].std(),
                    losses_clean2[~pred2].mean(),
                    losses_clean2[~pred2].std(),
                )
            )
            loss_log.flush()
            loader.run("train", pred2, prob2)  # count metrics
        else:
            prob2, all_loss[1], losses_clean2, weights2_raw = eval_train(
                net2,
                eval_loader,
                CE,
                all_loss[1],
                epoch,
                2,
                device,
                r,
                stats_log,
                weight_mode,
                log_name,
                codivide_policy,
                codivide_log,
                p_threshold,
                num_class,
                figures_folder,
                enableLog,
                compute_entropy=True,
                mcbn_forward_passes=3,
                per_class_testing_accuracy=per_class_accuracy,
            )

            prob1, all_loss[0], losses_clean1, weights1_raw = eval_train(
                net1,
                eval_loader,
                CE,
                all_loss[0],
                epoch,
                1,
                device,
                r,
                stats_log,
                weight_mode,
                log_name,
                codivide_policy,
                codivide_log,
                p_threshold,
                num_class,
                figures_folder,
            )

            # Updating weight history
            weight_hist_1[1:] = weight_hist_1[:-1]
            weight_hist_1[0, :] = weights1_raw
            weight_hist_2[1:] = weight_hist_2[:-1]
            weight_hist_2[0, :] = weights2_raw

            if epoch < (warm_up + window_size):
                weights1_smooth = weight_smoothing(
                    weight_hist_1[: epoch - warm_up],
                    num_class,
                    lambda_w_eps,
                    window_mode=window_mode,
                )
                weights2_smooth = weight_smoothing(
                    weight_hist_2[: epoch - warm_up],
                    num_class,
                    lambda_w_eps,
                    window_mode=window_mode,
                )
            else:
                weights1_smooth = weight_smoothing(
                    weight_hist_1, num_class, lambda_w_eps, window_mode=window_mode
                )
                weights2_smooth = weight_smoothing(
                    weight_hist_2, num_class, lambda_w_eps, window_mode=window_mode
                )
            # Write the weights to file
            # creating a csv writer object
            csvwriter = csv.writer(weights_log)
            # print([epoch])
            # print(weights1_raw)
            # print(weights2_raw)
            # print("(1) The type of weights1_raw is : ", type(weights1_raw))
            # print("(1) The size of weights1_raw is : ", len(weights1_raw))

            # print("(2) The type of weights1_smooth is : ", type(weights1_smooth))
            # print("(2) The size of weights1_smooth is : ", len(weights1_smooth))

            csvwriter.writerow([epoch] + weights1_raw.tolist() + weights2_raw.tolist())
            csvwriter.writerow([epoch] + weights1_smooth + weights2_smooth)
            weights_log.flush()

            print("\nTrain Net1")
            p_thr2 = np.clip(p_threshold, prob2.min() + 1e-5, prob2.max() - 1e-5)
            pred2 = prob2 > p_thr2

            loss_log.write(
                "{},{},{},{},{}\n".format(
                    epoch,
                    losses_clean2[pred2].mean(),
                    losses_clean2[pred2].std(),
                    losses_clean2[~pred2].mean(),
                    losses_clean2[~pred2].std(),
                )
            )
            loss_log.flush()

            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred2, prob2
            )  # co-divide
            train(
                epoch,
                net1,
                net2,
                criterion,
                optimizer1,
                labeled_trainloader,
                unlabeled_trainloader,
                lambda_u,
                batch_size,
                num_class,
                device,
                T,
                alpha,
                warm_up,
                dataset,
                r,
                noise_mode,
                num_epochs,
                weights1_smooth,
                training_losses_log,
                weightsLu,
                weightsLr,
            )  # train net1

            print("\nTrain Net2")
            # prob1, all_loss[0], losses_clean1, weights1_raw = eval_train(net1, eval_loader, CE, all_loss[0], epoch, 1,
            #                                               device, r, stats_log)

            p_thr1 = np.clip(p_threshold, prob1.min() + 1e-5, prob1.max() - 1e-5)
            pred1 = prob1 > p_thr1

            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred1, prob1
            )  # co-divide
            train(
                epoch,
                net2,
                net1,
                criterion,
                optimizer2,
                labeled_trainloader,
                unlabeled_trainloader,
                lambda_u,
                batch_size,
                num_class,
                device,
                T,
                alpha,
                warm_up,
                dataset,
                r,
                noise_mode,
                num_epochs,
                weights2_smooth,
                training_losses_log,
                weightsLu,
                weightsLr,
            )  # train net2

            if model_checkpoint_folder and (not epoch % 5 or epoch == 9):
                print(
                    f"[ SAVING MODELS] EPOCH: {epoch} PATH: {model_checkpoint_folder}"
                )
                save_net_optimizer_to_ckpt(
                    net1, optimizer1, f"{model_checkpoint_folder}/last_1.pt"
                )
                save_net_optimizer_to_ckpt(
                    net2, optimizer2, f"{model_checkpoint_folder}/last_2.pt"
                )

        per_class_accuracy = run_test(
            epoch, net1, net2, test_loader, device, test_log, num_class
        )

        sched1.step()
        sched2.step()
    final_checkpoint_name = (
        "./final_checkpoints/%s_final_checkpoint.pth.tar" % experiment_name
    )
    torch.save(net1.state_dict(), final_checkpoint_name)
