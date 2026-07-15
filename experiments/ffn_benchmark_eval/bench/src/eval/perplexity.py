"""
Perplexity (PPL) evaluation for language models.

Works with any tokenized dataset in the calibration format
{"input_ids": LongTensor [N, L], "attention_mask": LongTensor [N, L]}.

No dependency on the `datasets` library.

Usage:
    from src.eval.perplexity import eval_ppl

    ppl = eval_ppl(model, tokenized_data, batch_size=8)
    print(f"PPL = {ppl:.2f}")
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import tqdm

log = logging.getLogger(__name__)


@torch.no_grad()
def eval_ppl(
    model: nn.Module,
    data: dict,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> float:
    """Compute perplexity over a tokenized dataset.

    Uses manual CrossEntropyLoss(reduction="sum") to correctly aggregate NLL
    across batches, avoiding double-normalization issues with HF's loss output.

    Args:
        model:      AutoModelForCausalLM (eval mode, already on target device).
        data:       Dict with "input_ids" [N, L] and "attention_mask" [N, L].
        batch_size: Inference batch size.
        device:     Target device; inferred from model parameters if None.

    Returns:
        Perplexity as a float.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids: torch.Tensor = data["input_ids"]        # [N, L]
    attention_mask: torch.Tensor = data["attention_mask"]  # [N, L]
    n_samples = input_ids.shape[0]

    total_nll = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdm.tqdm(range(n_batches), desc="eval_ppl"):
        ids  = input_ids [i * batch_size : (i + 1) * batch_size].to(device)
        mask = attention_mask[i * batch_size : (i + 1) * batch_size].to(device)

        # Teacher-forcing: labels = input shifted left by 1
        labels = ids.clone()
        labels[mask == 0] = -100   # ignore padding tokens in loss

        # Get logits and compute loss manually to control reduction
        outputs = model(input_ids=ids, attention_mask=mask)
        logits = outputs.logits  # [batch, seq, vocab]

        # Shift logits and labels for causal LM loss
        shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq-1, vocab]
        shift_labels = labels[:, 1:].contiguous()      # [batch, seq-1]

        # Flatten for loss computation
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))  # [batch*(seq-1), vocab]
        shift_labels_flat = shift_labels.view(-1)                         # [batch*(seq-1)]

        # Only compute loss on valid tokens (not -100)
        valid_mask = shift_labels_flat != -100
        if valid_mask.sum() == 0:
            continue

        nll = loss_fn(shift_logits_flat[valid_mask], shift_labels_flat[valid_mask])
        total_nll += nll.double()
        total_tokens += int(valid_mask.sum().item())

    if total_tokens == 0:
        raise ValueError("No tokens to evaluate — all sequences may be fully padded.")

    mean_nll = total_nll / total_tokens
    ppl = float(torch.exp(mean_nll).item())
    log.info(f"PPL = {ppl:.4f}  (tokens={total_tokens})")
    return ppl


@torch.no_grad()
def eval_ppl_sliding_window(
    model: nn.Module,
    data: dict,
    batch_size: int = 4,
    stride: Optional[int] = None,
    device: Optional[str] = None,
    window_size: Optional[int] = None,
) -> float:
    """PPL with a sliding window — more accurate for long-context models.

    Each position is predicted using as much left context as possible
    (up to ``window_size``).  This matches the standard Wikitext evaluation
    protocol used in most pruning papers.

    The data for this protocol is a single ``[1, N]`` continuous token stream
    where N can be hundreds of thousands of tokens.  ``window_size`` is the
    per-forward context length (e.g. 2048) — it MUST come from the suite, NOT
    from ``input_ids.shape[1]`` (which is the whole stream length N; using it
    as the window would feed the entire stream into one forward pass and OOM).

    Args:
        stride:      Window stride; defaults to window_size // 2.
        window_size: Per-forward context length.  Defaults to min(2048, N) if
                     not given (back-compat), but callers should pass the
                     suite's seq_len explicitly.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids: torch.Tensor = data["input_ids"]
    attention_mask: torch.Tensor = data["attention_mask"]
    n_samples, stream_len = input_ids.shape

    # Window size = per-forward context, NOT the stream length.
    if window_size is None:
        window_size = min(2048, stream_len)
    seq_len = window_size

    if stride is None:
        stride = seq_len // 2

    total_nll = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    # ── 1) Pre-compute every window deterministically ────────────────────────
    # The serial version threads `prev_end` so each window scores only its
    # newly-revealed tokens (no double-count on the `stride` overlap).  That
    # sequence is fully determined by (real_len, seq_len, stride), so we can
    # enumerate all windows up front and then run them in parallel batches —
    # numerically identical to the serial loop, just GPU-batched.
    #
    # Each plan entry: (row_index, begin, end, score_start)
    #   - chunk = ids[begin:end]                 (context fed to the model)
    #   - only tokens [score_start, end) are scored (the new, non-overlapping part)
    plan = []
    for i in range(n_samples):
        mask = attention_mask[i]
        real_len = int(mask.sum().item())
        if real_len < 2:
            continue
        prev_end = 0
        for begin in range(0, real_len, stride):
            end = min(begin + seq_len, real_len)
            score_start = max(begin, prev_end)
            if end - score_start > 0:
                plan.append((i, begin, end, score_start))
            prev_end = end

    if not plan:
        raise ValueError("No tokens to evaluate.")

    # ── 2) Run windows in parallel batches, GROUPED BY LENGTH ────────────────
    # Almost every window is exactly `seq_len` long; only the last window of a
    # stream is shorter.  Grouping windows by length lets us batch same-length
    # windows with NO padding at all — so we avoid left-padding entirely and the
    # position-id / attention-mask subtleties it brings on real causal models.
    # The few short tail-windows form their own tiny groups.
    from collections import defaultdict
    by_len = defaultdict(list)
    for entry in plan:
        i, begin, end, score_start = entry
        by_len[end - begin].append(entry)

    n_windows = len(plan)
    pbar = tqdm.tqdm(total=n_windows, desc="eval_ppl_sliding")
    for clen, entries in by_len.items():
        for b0 in range(0, len(entries), batch_size):
            batch = entries[b0:b0 + batch_size]
            # All chunks here have identical length `clen` → plain stack, no pad.
            inp = torch.stack([input_ids[i, begin:end] for (i, begin, end, _) in batch]).to(device)
            with torch.no_grad():
                out = model(input_ids=inp)
            logits = out.logits  # [bsz, clen, vocab]

            for r, (i, begin, end, score_start) in enumerate(batch):
                # Serial parity: score global tokens [score_start+1, end), each
                # predicted from the logits at the previous position.
                lbl_start = score_start + 1
                if lbl_start >= end:
                    pbar.update(1)
                    continue
                off = score_start - begin            # within-window index of score_start
                n = end - lbl_start                  # tokens to score
                shift_logits = logits[r, off:off + n].contiguous()
                labels = input_ids[i, lbl_start:end].to(device)
                nll = loss_fn(shift_logits.float(), labels)
                total_nll += nll.double()
                total_tokens += labels.numel()
                pbar.update(1)
    pbar.close()

    if total_tokens == 0:
        raise ValueError("No tokens to evaluate.")

    ppl = float(torch.exp(total_nll / total_tokens).item())
    log.info(f"PPL (sliding) = {ppl:.4f}  (tokens={total_tokens}, batch_size={batch_size})")
    return ppl
