import numpy as np
from sklearn.mixture import GaussianMixture
import torch

def gmm_probabilities(loss, stats_log, epoch, net, targets=None):
    """
        To compute the GMM probabilities and log the means, weights and covariances.
    """
    gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss)

    clean_idx, noisy_idx = gmm.means_.argmin(), gmm.means_.argmax()
    stats_log.write('Epoch {} (net {}): GMM results: {} with weight {} and variance {} \t'
                    '{} with weight {} and variance {}\n'.format(epoch, net, 
                                    gmm.means_[clean_idx], gmm.weights_[clean_idx], gmm.covariances_[clean_idx],
                                    gmm.means_[noisy_idx], gmm.weights_[noisy_idx], gmm.covariances_[noisy_idx]))
    stats_log.flush()

    prob = gmm.predict_proba(loss)
    prob = prob[:, clean_idx]
    return prob


def ccgmm_probabilities(loss, stats_log, epoch, net, targets):
    """
        To compute the GMM probabilities with a Class-Conditional approach. And to log the means of the resulting GMM.
    """
    num_classes = int(torch.max(targets).item()+1)  # Find total number of classes
    prob = np.zeros(loss.size()[0])
    clean_means = 0
    noisy_means = 0

    for c in range(num_classes):
        mask = (targets == c).cpu().numpy()
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

