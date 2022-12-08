from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

from .ft_hparams import FTHyperParams
from rome.compute_v import find_fact_lookup_idx
from rome.rome_main import get_context_templates


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            request["target_new"]["str"] = " " + request["target_new"]["str"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    
    # MV: there's only 1 layer in hparams.layers for FT; prob because they dont weight decay and so it is not really used
    """
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    # MV: weights_copy (aka old_weights) is used to compute weight decay upon the diff (new_weights - old_weights)
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")
    """
    weights = [p for p in model.parameters()]
    weights_copy = [p.detach.clone() for p in model.parameters()]

    # Define inputs
    texts = [r["prompt"].format(r["subject"]) for r in requests]
    targets = [r["target_new"]["str"] for r in requests]

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights


    ### MV: adding KL stuff
    # Compile list of rewriting and KL x/y pairs
    # Tokenize target into list of int token IDs
    txt_to_info = dict()
    for i, txt in enumerate(texts):
        request = requests[i]
        target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
            "input_ids"
        ][0]
        ctlp =  [[5, 10], [10, 10]]
        if hasattr(hparams, "context_template_length_params"):
            ctlp = hparams.context_template_length_params
        context_templates = get_context_templates(model, tok, ctlp)
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context in context_templates
        ], ["{} is a"]
        all_prompts = rewriting_prompts + kl_prompts

        input_tok = tok(
            [prompt.format(request["subject"]) for prompt in all_prompts],
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        # Compute rewriting targets
        rewriting_targets = torch.tensor(-100, device="cuda").repeat(
            len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
        )
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

        # Compute indices of the tokens where the fact is looked up
        fact_token = hparams.fact_token if hasattr(hparams, "fact_token") else "subject_last"
        lookup_idxs = [
            find_fact_lookup_idx(
                prompt, request["subject"], tok, fact_token, verbose=(i == 0)
            )
            for i, prompt in enumerate(all_prompts)
        ]
        kl_distr_init = None
        txt_to_info[txt] = {
            "kl_distr_init" : kl_distr_init,
            "kl_prompts" : kl_prompts,
            "input_tok" : input_tok,
            "rewriting_targets" : rewriting_targets,
            "lookup_idxs" : lookup_idxs,
            "target_ids" : target_ids,
        }
    ### MV: end KL stuff


    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, 1), chunks(targets, 1) # MV: gotta do batch size of 1 for simplicity (anyhow we now submit more than 1 prompt because of the KL ones)
        ):
            opt.zero_grad()

            input_tok = txt_to_info[txt]["input_tok"]
            logits = model(**input_tok).logits

            # MV: get KL logits
            kl_logits = torch.stack(
                [
                    logits[i - len(txt_to_info[txt]["kl_prompts"]), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(txt_to_info[txt]["kl_prompts"]) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if txt_to_info[txt]["kl_distr_init"] is None:
                txt_to_info[txt]["kl_distr_init"] = kl_log_probs.detach().clone() # MV: will be set the first time, only for the KL prompts


            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)

            loss = torch.gather(
                log_probs,
                2,
                torch.where(txt_to_info[txt]["rewriting_targets"] != -100, txt_to_info[txt]["rewriting_targets"], 0).unsqueeze(2),
            ).squeeze(2)
            mask = (txt_to_info[txt]["rewriting_targets"] != -100).float()

            nll_loss_each = -(loss * mask).sum(1) / txt_to_info[txt]["target_ids"].size(0)
            nll_loss = nll_loss_each.mean()
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                txt_to_info[txt]["kl_distr_init"], kl_log_probs, log_target=True, reduction="batchmean"
            )

            loss_weight_decay = 0.0
            for k, _ in enumerate(weights):
                delta = weights[k] - weights_copy[k]
                loss_weight_decay += delta / torch.norm(weights_copy[k]) ** 2
            deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
            loss = nll_loss + kl_loss + loss_weight_decay
            ### MV: end edit

            loss = loss.mean()
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=1)

            loss.backward()
            opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        #if loss_meter.avg < 1e-2: # MV: ? why
        #    break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
