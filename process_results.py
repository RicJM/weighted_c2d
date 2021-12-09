import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process Learning Results C2D + Weights')
parser.add_argument('--noise', type=float, required=True)
parser.add_argument('--noise_type', type=str, default='sym')
parser.add_argument('--l_u', type=int, required=True)
parser.add_argument('--set', type=str, default='CF100')
parser.add_argument('--tau', type=float)
args   = parser.parse_args()

noise_level = args.noise
if args.set == 'CF100':
    
    if args.tau is None:
        file_names  = './checkpoint/%s_%s_*_%.2f_%.1f_%s_acc.txt' % (args.set, args.noise_type, noise_level, float(args.l_u), args.noise_type)
    else:
        file_names  = './checkpoint/%s_%s_*_tau_%.2f_*_%.2f_%.1f_%s_acc.txt' % (args.set, args.noise_type, args.tau, noise_level, float(args.l_u), args.noise_type)

elif args.set == 'CF10':
    file_names  = './checkpoint/%s_*_%.2f_%.1f_%s_acc.txt' % (args.noise_type, noise_level, float(args.l_u), args.noise_type)
else:
    print('Unknown dataset')

print(file_names)
file_list   = glob.glob(file_names)

print('\n')
best_acc_list   = []
final_acc_list  = []
counter = 0
for i, f_n in enumerate(file_list):
    print(f_n)
    with open(f_n) as f:
        best_acc   = 0;
        best_epoch = -1;
        accuracy   = 0;
        for l, line in enumerate(f):
            str_test = line.split()
            epoch    = int(str_test[0].split(":")[1])
            accuracy = float(str_test[1].split(":")[1])
            if accuracy >= best_acc:
                best_acc   = accuracy
                best_epoch = epoch
        final_acc  = accuracy
        last_epoch = l
    print("\tbest epoch: %d; best accuracy: %.2f" % (best_epoch, best_acc))
    print("\tlast epoch: %d; last accuracy: %.2f\n" % (last_epoch, final_acc))

    if last_epoch >= 359:
        counter += 1
        best_acc_list.append(best_acc)
        final_acc_list.append(final_acc)

print('Number of finished runs %d\n' % counter)
if counter > 0:
    print('Average Best Accuracy: %.3f' % np.average(best_acc_list))
    print('Standard Deviation Best Accuracy: %.3f\n' % np.std(best_acc_list))
    print('Average Last Accuracy: %.3f' % np.average(final_acc_list))
    print('Standard Deviation Last Accuracy: %.3f\n' % np.std(final_acc_list))
