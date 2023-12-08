from collections import defaultdict

import baukit as nethook
import numpy
import torch

from causal_trace.utils import (
    decode_tokens,
    find_token_range,
    make_inputs,
    predict_from_input,
)


def trace_with_patch(
    model,
    embed_layername,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def trace_with_repatch(
    model,
    embed_layername,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    # TODO fix the noise thing. base on the model.
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    if expect is not None and answer.strip() != expect:
        return dict(correct_prediction=False)
    e_range = find_token_range(
        string=prompt,
        substring=subject,
        tokenizer=mt.tokenizer,
    )
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    low_score = trace_with_patch(
        mt.model,
        mt.embedder_name,
        inp,
        [],
        answer_t,
        e_range,
        noise=noise,
        uniform_noise=uniform_noise,
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model,
            mt.embedder_name,
            mt.n_layer,
            mt.layer_name_format,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        module_name_format = (
            mt.mlp_module_name_format if kind == "mlp" else mt.attn_module_name_format
        )

        differences = trace_important_window(
            mt.model,
            mt.embedder_name,
            mt.n_layer,
            module_name_format,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            token_range=token_range,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )


def trace_important_states(
    model,
    embed_layername,
    num_layers,
    layer_name_format,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                embed_layername,
                inp,
                [(tnum, layer_name_format.format(layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model,
    embed_layername,
    num_layers,
    module_name_format,
    inp,
    e_range,
    answer_t,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, module_name_format.format(L))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model,
                embed_layername,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)
