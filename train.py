import sys

import numpy as np
import torch
import torch.nn.functional as F
import gc


def co_guess(
    net, net2, inputs_x, inputs_u, inputs_x2, inputs_u2, w_x, labels_x, T, smooth_clean
):
    # label co-guessing of unlabeled samples
    outputs_u11 = net(inputs_u)
    outputs_u12 = net(inputs_u2)
    outputs_u21 = net2(inputs_u)
    outputs_u22 = net2(inputs_u2)

    pu = (
        torch.softmax(outputs_u11, dim=1)
        + torch.softmax(outputs_u12, dim=1)
        + torch.softmax(outputs_u21, dim=1)
        + torch.softmax(outputs_u22, dim=1)
    ) / 4
    ptu = pu ** (1 / T)  # temperature sharpening

    targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
    targets_u = targets_u.detach()

    # label refinement of labeled samples
    outputs_x = net(inputs_x)
    outputs_x2 = net(inputs_x2)

    if smooth_clean:
        px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
        px = w_x * labels_x + (1 - w_x) * px
        ptx = px ** (1 / T)  # temparature sharpening

        targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
        targets_x = targets_x.detach()
    else:
        targets_x = labels_x
    return targets_x, targets_u


# Training
def train(
    epoch,
    net,
    net2,
    criterion,
    optimizer,
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
    smooth_clean=True,
):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, _, w_x) in enumerate(
        labeled_trainloader
    ):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, num_class).scatter_(
            1, labels_x.view(-1, 1), 1
        )
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2 = inputs_x.to(device), inputs_x2.to(device)
        labels_x, w_x = labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            targets_x, targets_u = co_guess(
                net,
                net2,
                inputs_x,
                inputs_u,
                inputs_x2,
                inputs_u2,
                w_x,
                labels_x,
                T,
                smooth_clean,
            )

        # mixmatch
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        if lambda_u > 0:
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net(mixed_input)
            logits_x = logits[: batch_size * 2]
            logits_u = logits[batch_size * 2 :]

            Lx, Lu, lamb = criterion(
                logits_x,
                mixed_target[: batch_size * 2],
                logits_u,
                mixed_target[batch_size * 2 :],
                epoch + batch_idx / num_iter,
                warm_up,
                lambda_u,
            )
        else:
            mixed_input = (
                l * input_a[: batch_size * 2] + (1 - l) * input_b[: batch_size * 2]
            )
            mixed_target = (
                l * target_a[: batch_size * 2] + (1 - l) * target_b[: batch_size * 2]
            )

            logits = net(mixed_input)

            Lx = -torch.mean(
                torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1)
            )
            lamb, Lu = 0, 0
        # regularization
        prior = torch.ones(num_class) / num_class
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write("\r")
        if "cifar" in "dataset":
            sys.stdout.write(
                "%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t"
                "Labeled loss: %.2f  Unlabeled loss: %.2e(%.2e)  penalty: %.2e"
                % (
                    dataset,
                    r,
                    noise_mode,
                    epoch,
                    num_epochs,
                    batch_idx + 1,
                    num_iter,
                    Lx.item(),
                    Lu.item(),
                    lamb * Lu.item(),
                    penalty.item(),
                )
            )
        elif "clothing" in dataset:
            sys.stdout.write(
                "Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t"
                "Labeled loss: %.2f  penalty: %.2e"
                % (
                    epoch,
                    num_epochs,
                    batch_idx + 1,
                    num_iter,
                    Lx.item(),
                    penalty.item(),
                )
            )
        sys.stdout.flush()


def bytes2human(n, format="%(value).1f %(symbol)s", symbols="customary"):
    SYMBOLS = {
        "customary": ("B", "K", "M", "G", "T", "P", "E", "Z", "Y"),
        "customary_ext": (
            "byte",
            "kilo",
            "mega",
            "giga",
            "tera",
            "peta",
            "exa",
            "zetta",
            "iotta",
        ),
        "iec": ("Bi", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"),
        "iec_ext": (
            "byte",
            "kibi",
            "mebi",
            "gibi",
            "tebi",
            "pebi",
            "exbi",
            "zebi",
            "yobi",
        ),
    }
    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


def warmup(
    epoch,
    net,
    optimizer,
    dataloader,
    criterion,
    conf_penalty,
    device,
    dataset,
    r,
    num_epochs,
    noise_mode,
):
    net.train()
    for batch_idx, (inputs, _, labels, _, _) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        # print(f"{batch_idx} {bytes2human(torch.cuda.memory_allocated(device=None))}")
        # print(inputs.size())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        assert torch.isfinite(loss).all()
        penalty = conf_penalty(outputs) if conf_penalty is not None else 0.0
        L = loss + penalty
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        sys.stdout.write("\r")
        if "clothing" in dataset:
            sys.stdout.write(
                "|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f"
                % (batch_idx + 1, len(dataloader), loss.item(), penalty.item())
            )
        elif "cifar" in dataset:
            sys.stdout.write(
                "%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f"
                % (
                    dataset,
                    r,
                    noise_mode,
                    epoch,
                    num_epochs,
                    batch_idx + 1,
                    len(dataloader),
                    loss.item(),
                )
            )
        sys.stdout.flush()
