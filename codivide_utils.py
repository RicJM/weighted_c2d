from json.tool import main
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import matplotlib.pyplot as plt

def gmm_probabilities(loss, stats_log, epoch, net, targets=None, log=True):
    """
        To compute the GMM probabilities and log the means, weights and covariances.
    """
    gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss)

    clean_idx, noisy_idx = gmm.means_.argmin(), gmm.means_.argmax()
    
    if log:
        stats_log.write('Epoch {} (net {}): GMM results: {} with weight {} and variance {} \t'
                        '{} with weight {} and variance {}\n'.format(epoch, net, 
                                        gmm.means_[clean_idx], gmm.weights_[clean_idx], gmm.covariances_[clean_idx],
                                        gmm.means_[noisy_idx], gmm.weights_[noisy_idx], gmm.covariances_[noisy_idx]))
        stats_log.flush()

    prob = gmm.predict_proba(loss)
    prob = prob[:, clean_idx]
    return prob

def ccgmm_probabilities(loss, stats_log, epoch, net, targets, log=True):
    """
        To compute the GMM probabilities with a Class-Conditional approach. And to log the means of the resulting GMM.
        This function also computes the original GMM division and logs the comparison between the two.

        @params:
        - targets - np.array with the class of every element.
    """
    num_classes = max(targets)+1  # Find total number of classes
    prob = np.zeros(loss.size()[0])
    clean_means = 0
    noisy_means = 0

    for c in range(num_classes):
        mask = targets == c

        gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss[:,0][mask].reshape(-1,1))

        clean_idx, noisy_idx = gmm.means_.argmin(), gmm.means_.argmax()
        clean_means += gmm.means_[clean_idx]
        noisy_means += gmm.means_[noisy_idx]

        p = gmm.predict_proba(loss[:,0][mask].reshape(-1,1))
        prob[mask] = p[:, clean_idx]


    stats_log.write('Epoch {} (net {}): GMM results: {} with weight {} and variance {} \t'
                    '{} with weight {} and variance {}\n'.format(epoch, net, 
                                    clean_means/num_classes, 'not_recorded', 'not_recorded',
                                    noisy_means/num_classes, 'not_recorded', 'not_recorded')) # GMM means will be the mean of the gmm means for each class.
    stats_log.flush()

    return prob

def codivide_gmm(loss, stats_log, epoch, net, 
                p_threshold, targets, targets_clean, codivide_log):
    '''
        Computes co-divide following two different policies to print/log the comparison between the two.
        The main_policy is used to follow the execution and the benchmark_policy is discarded.
    '''
    prob = gmm_probabilities(loss, stats_log, epoch, net, targets)
    prob_benchmark = ccgmm_probabilities(loss, stats_log, epoch, net, targets, log=False)
    clean_samples = targets_clean==targets
    probs = [prob, prob_benchmark]
    policy_names = ['GMM', 'CCGMM']

    results = [benchmark(prob, name, p_threshold, targets, clean_samples) for prob, name in list(zip(probs, policy_names))]
    string=''.join(results)
    print(f'{string}')
    codivide_log.write(string)
    codivide_log.flush()

    return prob

def codivide_ccgmm(loss, stats_log, epoch, net, 
                p_threshold, targets, targets_clean, codivide_log):
    '''
        Computes co-divide following two different policies to print/log the comparison between the two.
        The main_policy is used to follow the execution and the benchmark_policy is discarded.
    '''
    prob = ccgmm_probabilities(loss, stats_log, epoch, net, targets)
    prob_benchmark = gmm_probabilities(loss, stats_log, epoch, net, targets, log=False)

    clean_samples = targets_clean==targets
    probs = [prob, prob_benchmark]
    policy_names = ['CCGMM', 'GMM']

    results = [benchmark(prob, name, p_threshold, targets, clean_samples) for prob, name in list(zip(probs, policy_names))]
    string=''.join(results)
    print(f'{string}')
    codivide_log.write(string)
    codivide_log.flush()

    return prob

def benchmark(prob, name, p_threshold, targets, clean_samples):
    """
        To benchmark a given probability list.
    """
    p_thr = np.clip(p_threshold, prob.min() + 1e-5, prob.max() - 1e-5)
    pred = prob > p_thr
    
    comparison = clean_samples == pred
    tp = np.logical_and(comparison, clean_samples)
    tn = np.logical_and(comparison, ~clean_samples)
    fp = np.logical_and(~comparison, ~clean_samples)
    fn = np.logical_and(~comparison, clean_samples)
    precision = tp.sum()/(fp.sum()+fp.sum())
    recall = tp.sum()/(tp.sum()+fn.sum())
    f1_score = recall*precision/(precision+recall)
    accuracy = comparison.sum()/len(comparison)
    std = np.std([comparison[targets==c].sum() for c in range(max(targets)+1)]) # sum number of correct predictions for each class
    return f'\t{name} Accuracy:{accuracy:.3f} std:{std:.3f} f1_score:{f1_score:.3f} fp:{sum(fp)/len(comparison):.3f} fn:{sum(fn)/len(comparison):.3f}\n'

def per_sample_plot(loss, targets, targets_clean, figures_folder, epoch):
    CIFAR10_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    CIFAR100_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    dispersion = np.random.rand(len(targets_clean))*0.8
    clean_labels = targets == targets_clean
    s=0.2
    num_classes = max(targets_clean)+1
    readable_labels = CIFAR10_labels if num_classes==10 else CIFAR100_labels
    plt.figure(figsize=(35, 10), dpi=80)
    for c in range(0, num_classes):
        class_mask = targets_clean==c
        plt.scatter(dispersion[clean_labels&class_mask]+2*c, loss[clean_labels&class_mask], c='blue', marker='.', s=s)
        plt.scatter(dispersion[(~clean_labels)&class_mask]+(0.8+2*c),loss[(~clean_labels)&class_mask], c='red', marker='.', s=s)
    plt.xticks(range(0,num_classes*2,2), readable_labels, rotation=90)
    plt.xlabel('Label')
    plt.ylabel('Loss')
    plt.title(f'Per-sample loss distribution\nRed noisy | Rlue clean\nEpoch: {epoch}')
    plt.savefig(f'{figures_folder}/{epoch}.png')