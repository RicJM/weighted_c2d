from json.tool import main
import numpy as np
from sklearn.mixture import GaussianMixture
import torch

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
    benchmark_policies([prob, prob_benchmark], ['GMM', 'CCGMM'], p_threshold, targets, clean_samples, codivide_log)

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
    benchmark_policies([prob, prob_benchmark], ['CCGMM', 'GMM'], p_threshold, targets, clean_samples, codivide_log)

    return prob

def benchmark_policies(probs, policy_names, p_threshold, targets, clean_samples, codivide_log):
    """
        To compute the accuracy, F1 score, %FP, %FN and Standard distribution across the classes of the different probability vectors.
        @params:
            probs: list of np.arrays containing all the probs for each policy to benchmark.        
    """

    results = [benchmark(prob, name, p_threshold, targets, clean_samples) for prob, name in list(zip(probs, policy_names))]
    string=' '.join(results)
    print(f'Co-Didivide Benchmark\n{string}')
    codivide_log.write(string)
    codivide_log.flush()


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
    std = np.std([comparison[targets==c] for c in range(max(targets)+1)])
    return f'Policy:{name} Accuracy:{accuracy} std:{std} f1_score:{f1_score} fp:{fp.sum()/len(comparison)} fn:{fn/len(comparison)}\n'
