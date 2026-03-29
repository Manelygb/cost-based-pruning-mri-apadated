# finetune.py (replacement)
import os
import os.path as osp

import torch
import numpy as np
from tqdm import tqdm

from utils import batch, AverageMeter, get_imgs_and_masks
from flops_counter import flops_count

def _ensure_trailing_sep(p: str) -> str:
    p = osp.expanduser(p)
    return p if p.endswith(os.sep) else p + os.sep

def _resolve_data_dirs() -> tuple[str, str]:
    return (
        _ensure_trailing_sep("dataset_split/train/imgs"),
        _ensure_trailing_sep("dataset_split/train/masks"),
    )

def finetune(net, optimizer, criterion, trainset, log, path,
             iters=100, epochs=None, batch_size=2, gpu=True, scale=0.5,
             # --- val args (optional) ---
             valset=None, dir_img_val=None, dir_mask_val=None, eval_fn=None):
    """
    Finetune net.

    If valset + dir_img_val + dir_mask_val + eval_fn are all provided,
    val Dice is computed after every epoch and the best checkpoint is saved.
    Otherwise behaviour is identical to the original (save last epoch).
    """
    net.train()
    loss_meter = AverageMeter()

    dir_img, dir_mask = _resolve_data_dirs()
    log.info(f"[finetune] Using dir_img={dir_img} dir_mask={dir_mask}")

    use_val = (valset is not None and dir_img_val is not None
               and dir_mask_val is not None and eval_fn is not None)

    if use_val:
        log.info(f"[finetune] Val-based best-checkpoint saving ENABLED ({len(valset)} val samples)")
    else:
        log.info("[finetune] Val args not provided — will save last-epoch checkpoint (original behaviour)")

    if epochs is None:  # Fine-tune using iterations of mini-batches
        epochs = 1
    else:               # Fine-tune using entire epochs
        iters = None

    best_dice = -1.0
    best_path = path + ".best.pth"   # temp file; renamed to `path` at the end

    for e in range(epochs):
        net.train()
        loss_meter = AverageMeter()

        # reset the generators
        train = get_imgs_and_masks(trainset, dir_img, dir_mask, scale)

        with tqdm(total=len(trainset)) as progress_bar:
            for i, b in enumerate(batch(train, batch_size)):
                imgs = np.array([i0[0] for i0 in b]).astype(np.float32)
                true_masks = np.array([i0[1] for i0 in b])

                imgs = torch.from_numpy(imgs)                  # [B,3,H,W] float32
                true_masks = torch.from_numpy(true_masks)      # could be [B,1,H,W] or [B,H,W]

                # ---- Target formatting for CrossEntropyLoss ----
                if true_masks.ndim == 4 and true_masks.shape[1] == 1:
                    true_masks = true_masks[:, 0, :, :]        # [B,H,W]

                if true_masks.max() > 1:
                    true_masks = (true_masks > 0).long()
                else:
                    true_masks = true_masks.long()

                if gpu:
                    imgs = imgs.cuda(non_blocking=True)
                    true_masks = true_masks.cuda(non_blocking=True)

                # ---- Forward ----
                masks_pred = net(imgs)                         # [B,2,H,W] float logits

                if not masks_pred.is_floating_point():
                    masks_pred = masks_pred.float()

                loss = criterion(masks_pred, true_masks)
                loss_meter.update(loss.item(), imgs.shape[0])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                progress_bar.update(imgs.shape[0])
                progress_bar.set_postfix(epoch=e, loss=loss_meter.avg)

                if i == 0 and e == 0:
                    try:
                        log.info("FLOPs after pruning: \n{}".format(flops_count(net, imgs.shape[2:])))
                    except Exception as ex:
                        log.info(f"[finetune] FLOPs count failed: {ex}")

                if iters is not None and i >= iters:
                    break

        log.info(f"Epoch {e} finished | avg train loss: {loss_meter.avg:.6f}")

        # ---- Per-epoch val Dice + best checkpoint ----
        if use_val:
            val_gen = get_imgs_and_masks(valset, dir_img_val, dir_mask_val, scale)
            val_dice = eval_fn(net, val_gen, len(valset), gpu, batch_size)
            net.train()  # restore train mode after eval

            log.info(f"Epoch {e} | val Dice: {val_dice:.6f} | best so far: {best_dice:.6f}")

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(net.state_dict(), best_path)
                log.info(f"  → New best! Saved checkpoint (dice={best_dice:.6f})")

    log.info("Finished finetuning")
    log.info("Finetuned loss: {}".format(loss_meter.avg))

    if use_val:
        # Replace `path` with the best checkpoint
        if osp.exists(best_path):
            os.replace(best_path, path)
            log.info(f"Best val Dice={best_dice:.6f} — saved best checkpoint to {path}")
        else:
            # Fallback: best_path was never written (shouldn't happen)
            torch.save(net.state_dict(), path)
            log.info(f"Warning: best checkpoint file missing; saved last epoch to {path}")

        # Load best weights back into net so the caller evaluates the best model
        state = torch.load(path, map_location="cuda" if gpu else "cpu", weights_only=True)
        net.load_state_dict(state)
        log.info("Loaded best weights back into model")
    else:
        torch.save(net.state_dict(), path)
        log.info("Saving finetuned to {}...".format(path))