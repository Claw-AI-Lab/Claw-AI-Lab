import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch_npu  # MUST import before using torch.npu
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as torchvision_datasets

from models import (
    ResNetNoReg,
    ResNetDropout,
    ResNetSpatialDropout,
    ResNetDropBlock,
    ResNetDropPath,
    ResNetBatchNormOnly,
    ResNetL2Only,
    ResNetCORAL,
    ResNetAugMax,
    ResNetScheduledDropout,
    ResNetMCDropout,
)
from experiment_harness import ExperimentHarness

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HYPERPARAMETERS = {
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay_default': 5e-4,
    'weight_decay_l2': 5e-3,
    'batch_size': 128,
    'num_epochs': 60,
    'lr_milestones': [30, 45],
    'lr_gamma': 0.1,
    'dropout_p': 0.5,
    'dropblock_block_size': 7,
    'dropblock_drop_prob': 0.1,
    'droppath_drop_prob': 0.2,
    'coral_lambda': 0.5,
    'augmax_steps': 3,
    'augmax_step_size': 0.1,
    'scheduled_dropout_start': 0.0,
    'scheduled_dropout_end': 0.5,
    'mc_dropout_p': 0.3,
    'num_workers': 8,
    'val_fraction': 0.1,
    'seeds': [42, 123, 456],
    'ece_bins': 15,
}

DATASETS_DIR = '/var/lib/paascontainer/zty/Claw-AI-Lab-share/datasets'

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device('npu' if torch.npu.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_cifar10_loaders(seed, batch_size=128, num_workers=8, val_fraction=0.1):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    cifar10_root = os.path.join(DATASETS_DIR, 'cifar-10-batches-py')
    # torchvision expects the parent directory
    data_root = DATASETS_DIR

    full_train = torchvision_datasets.CIFAR10(
        root=data_root, train=True, transform=train_transform, download=False
    )
    full_train_eval = torchvision_datasets.CIFAR10(
        root=data_root, train=True, transform=eval_transform, download=False
    )
    test_dataset = torchvision_datasets.CIFAR10(
        root=data_root, train=False, transform=eval_transform, download=False
    )

    n_total = len(full_train)
    n_val   = int(np.floor(val_fraction * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )
    train_indices = list(train_indices)
    val_indices   = list(val_indices)

    train_subset = Subset(full_train,      train_indices)
    val_subset   = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader

# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------
def compute_ece(model, loader, n_bins=15, use_mc=False, mc_samples=10):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if use_mc:
                # MC dropout: average over multiple forward passes
                model.train()  # enable dropout
                preds = []
                for _ in range(mc_samples):
                    logits = model(images)
                    preds.append(torch.softmax(logits, dim=1))
                probs = torch.stack(preds).mean(0)
                model.eval()
            else:
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)

    all_probs  = torch.cat(all_probs,  dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    confidences = all_probs.max(axis=1)
    predictions = all_probs.argmax(axis=1)
    accuracies  = (predictions == all_labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_acc  = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    ece /= len(all_labels)
    return float(ece)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, use_mc=False, mc_samples=10):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if use_mc:
                model.train()
                preds = []
                for _ in range(mc_samples):
                    logits = model(images)
                    preds.append(torch.softmax(logits, dim=1))
                probs  = torch.stack(preds).mean(0)
                logits = torch.log(probs + 1e-8)
                model.eval()
            else:
                logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            _, pred_top5 = logits.topk(5, dim=1)
            labels_exp = labels.view(-1, 1).expand_as(pred_top5)
            correct_top5 += pred_top5.eq(labels_exp).any(dim=1).sum().item()
            correct_top1 += pred_top5[:, 0].eq(labels).sum().item()
            total += labels.size(0)

    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total
    avg_loss = total_loss / total
    return acc_top1, acc_top5, avg_loss

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, condition_name,
                    epoch, total_epochs, harness, augmax_model=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        if condition_name == 'AugMax' and augmax_model is not None:
            # AugMax: adversarially augment inputs
            images_01 = torch.clamp(images, 0.0, 1.0)
            images = augmax_model.augmax_augment(
                images_01,
                steps=HYPERPARAMETERS['augmax_steps'],
                step_size=HYPERPARAMETERS['augmax_step_size'],
                labels=labels,
            )
            model.train()

        if condition_name == 'CORAL':
            # Use same batch as source and a shuffled version as target
            idx = torch.randperm(images.size(0))
            x_target = images[idx]
            logits, coral_loss = model(images, x_target)
            cls_loss  = criterion(logits, labels)
            loss = cls_loss + HYPERPARAMETERS['coral_lambda'] * coral_loss
        elif condition_name == 'ScheduledDropout':
            # Update dropout rate based on epoch
            progress = epoch / total_epochs
            p_current = (HYPERPARAMETERS['scheduled_dropout_start'] +
                         progress * (HYPERPARAMETERS['scheduled_dropout_end'] -
                                     HYPERPARAMETERS['scheduled_dropout_start']))
            model.set_dropout_rate(p_current)
            logits = model(images)
            loss   = criterion(logits, labels)
        else:
            logits = model(images)
            loss   = criterion(logits, labels)

        if not harness.check_value(loss.item(), 'train_loss'):
            print('FAIL: NaN/divergence detected in training loss')
            return None, None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += logits.argmax(dim=1).eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total

def run_experiment(condition_name, model_fn, seed, num_epochs, harness):
    set_all_seeds(seed)
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        seed=seed,
        batch_size=HYPERPARAMETERS['batch_size'],
        num_workers=HYPERPARAMETERS['num_workers'],
        val_fraction=HYPERPARAMETERS['val_fraction'],
    )

    use_mc = (condition_name == 'MCDropout')

    # Build model
    wd = None
    if condition_name == 'L2Only':
        wd = HYPERPARAMETERS['weight_decay_l2']
    else:
        wd = HYPERPARAMETERS['weight_decay_default']

    model = model_fn().to(device)

    # For AugMax we also need the AugMax wrapper
    augmax_model = None
    if condition_name == 'AugMax':
        from models import ResNetAugMax as AugMaxCls
        augmax_model = AugMaxCls().to(device)
        # Share backbone weights
        augmax_model.backbone = model.backbone if hasattr(model, 'backbone') else model

    optimizer = optim.SGD(
        model.parameters(),
        lr=HYPERPARAMETERS['learning_rate'],
        momentum=HYPERPARAMETERS['momentum'],
        weight_decay=wd,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=HYPERPARAMETERS['lr_milestones'],
        gamma=HYPERPARAMETERS['lr_gamma'],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(num_epochs):
        if harness.should_stop():
            print(f"TIME_GUARD: stopping at epoch {epoch} for condition={condition_name} seed={seed}")
            break

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            condition_name, epoch, num_epochs, harness,
            augmax_model=augmax_model,
        )
        if train_loss is None:
            return None

        scheduler.step()

        val_acc, val_top5, val_loss = evaluate(model, val_loader,
                                                use_mc=use_mc, mc_samples=10)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  [condition={condition_name} seed={seed}] "
                  f"epoch={epoch+1}/{num_epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_acc={val_acc:.4f}")

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final evaluation
    test_acc_top1, test_acc_top5, test_loss = evaluate(
        model, test_loader, use_mc=use_mc, mc_samples=10
    )
    train_acc_final, _, _ = evaluate(model, train_loader)
    gen_gap = train_acc_final - test_acc_top1

    ece = compute_ece(model, test_loader, n_bins=HYPERPARAMETERS['ece_bins'],
                      use_mc=use_mc, mc_samples=10)

    # Validation loss as primary metric (lower is better)
    _, _, val_loss_final = evaluate(model, val_loader, use_mc=use_mc, mc_samples=10)

    return {
        'test_acc_top1':  test_acc_top1,
        'test_acc_top5':  test_acc_top5,
        'val_loss':       val_loss_final,
        'gen_gap':        gen_gap,
        'ece':            ece,
        'primary_metric': val_loss_final,  # minimize val_loss
    }

# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------
def build_conditions():
    from models import (
        ResNetNoReg, ResNetDropout, ResNetSpatialDropout,
        ResNetDropBlock, ResNetDropPath, ResNetBatchNormOnly,
        ResNetL2Only, ResNetCORAL, ResNetAugMax,
        ResNetScheduledDropout, ResNetMCDropout,
    )
    p = HYPERPARAMETERS['dropout_p']
    db_size = HYPERPARAMETERS['dropblock_block_size']
    db_prob = HYPERPARAMETERS['dropblock_drop_prob']
    dp_prob = HYPERPARAMETERS['droppath_drop_prob']
    mc_p    = HYPERPARAMETERS['mc_dropout_p']

    conditions = {
        'NoReg':           lambda: ResNetNoReg(),
        'L2Only':          lambda: ResNetL2Only(),
        'BNOnly':          lambda: ResNetBatchNormOnly(),
        'Dropout_p05':     lambda: ResNetDropout(dropout_p=p),
        'Dropout_p01':     lambda: ResNetDropout(dropout_p=0.1),
        'Dropout_p02':     lambda: ResNetDropout(dropout_p=0.2),
        'Dropout_p03':     lambda: ResNetDropout(dropout_p=0.3),
        'Dropout_p07':     lambda: ResNetDropout(dropout_p=0.7),
        'SpatialDropout':  lambda: ResNetSpatialDropout(dropout_p=p),
        'DropBlock':       lambda: ResNetDropBlock(block_size=db_size, drop_prob=db_prob),
        'DropPath':        lambda: ResNetDropPath(drop_prob=dp_prob),
        'MCDropout':       lambda: ResNetMCDropout(dropout_p=mc_p),
        'ScheduledDropout':lambda: ResNetScheduledDropout(
                                        start_p=HYPERPARAMETERS['scheduled_dropout_start'],
                                        end_p=HYPERPARAMETERS['scheduled_dropout_end']),
        'CORAL':           lambda: ResNetCORAL(),
        'AugMax':          lambda: ResNetAugMax(),
    }
    return conditions

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("METRIC_DEF: primary_metric | direction=lower | desc=Validation cross-entropy loss (lower is better)")

    conditions = build_conditions()
    print(f"REGISTERED_CONDITIONS: {', '.join(conditions.keys())}")

    SEEDS = HYPERPARAMETERS['seeds']
    print(f"SEEDS: {SEEDS}")
    print(f"NUM_EPOCHS: {HYPERPARAMETERS['num_epochs']}")

    harness = ExperimentHarness(time_budget=2400)

    # ---- Pilot timing ----
    pilot_start = time.time()
    set_all_seeds(42)
    pilot_model = build_conditions()['Dropout_p05']().to(device)
    tl, _, _ = get_cifar10_loaders(42, batch_size=128, num_workers=8)
    opt_p = optim.SGD(pilot_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    crit  = nn.CrossEntropyLoss()
    pilot_model.train()
    for imgs, lbls in tl:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt_p.zero_grad()
        loss = crit(pilot_model(imgs), lbls)
        loss.backward()
        opt_p.step()
        break  # single batch pilot
    pilot_epoch_time = time.time() - pilot_start
    batches_per_epoch = len(tl)
    epoch_time_est = pilot_epoch_time * batches_per_epoch
    total_est = epoch_time_est * HYPERPARAMETERS['num_epochs'] * len(conditions) * len(SEEDS)
    print(f"TIME_ESTIMATE: {total_est:.0f}s  (epoch≈{epoch_time_est:.1f}s × "
          f"{HYPERPARAMETERS['num_epochs']} epochs × {len(conditions)} conditions × {len(SEEDS)} seeds)")
    del pilot_model, opt_p

    # ---- Breadth-first: 1 seed first, then more ----
    all_results = {cname: {} for cname in conditions}
    collected_metrics = {}

    os.makedirs('outputs', exist_ok=True)

    for seed in SEEDS:
        if harness.should_stop():
            print(f"TIME_GUARD: stopping before seed={seed}")
            break
        print(f"\n{'='*60}")
        print(f"RUNNING SEED={seed}")
        print(f"{'='*60}")

        for cname, model_fn in conditions.items():
            if harness.should_stop():
                print(f"TIME_GUARD: stopping before condition={cname} seed={seed}")
                break

            print(f"\n--- condition={cname} seed={seed} ---")
            result = run_experiment(
                condition_name=cname,
                model_fn=model_fn,
                seed=seed,
                num_epochs=HYPERPARAMETERS['num_epochs'],
                harness=harness,
            )
            if result is None:
                print(f"SKIP: condition={cname} seed={seed} failed")
                continue

            pmetric = result['primary_metric']
            if not harness.check_value(pmetric, 'primary_metric'):
                print(f"SKIP: NaN/Inf primary_metric for condition={cname} seed={seed}")
                continue

            harness.report_metric('primary_metric', pmetric)
            all_results[cname][seed] = result

            print(f"condition={cname} seed={seed} primary_metric: {pmetric:.6f}")
            print(f"  test_acc_top1={result['test_acc_top1']:.4f} "
                  f"test_acc_top5={result['test_acc_top5']:.4f} "
                  f"gen_gap={result['gen_gap']:.4f} "
                  f"ece={result['ece']:.4f}")

            # Save intermediate
            intermediate = {cname: {str(s): v for s, v in all_results[cname].items()}}
            with open('outputs/intermediate_results.json', 'w') as f:
                json.dump(intermediate, f, indent=2)

    # ---- Aggregate per condition ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for cname in conditions:
        seed_results = all_results[cname]
        if not seed_results:
            continue
        vals = [v['primary_metric'] for v in seed_results.values()]
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        print(f"condition={cname} primary_metric_mean: {mean_v:.6f} primary_metric_std: {std_v:.6f}")
        collected_metrics[cname] = {
            'primary_metric_mean': mean_v,
            'primary_metric_std':  std_v,
            'seeds': {str(s): {k: float(v) for k, v in res.items()}
                      for s, res in seed_results.items()},
        }

    # ---- Save results.json ----
    results = {
        'hyperparameters': HYPERPARAMETERS,
        'metrics': collected_metrics,
        'conditions': {
            cname: {str(s): {k: float(v) for k, v in res.items()}
                    for s, res in seed_results.items()}
            for cname, seed_results in all_results.items()
            if seed_results
        },
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results.json")

    harness.finalize()
    print("DONE")

if __name__ == '__main__':
    main()