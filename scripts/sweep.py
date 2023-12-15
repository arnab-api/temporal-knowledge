import argparse
import logging
import os

from relations.src.operators import JacobianIclMeanEstimator
from src import functional
from src.dataset import TemporalRelation, balance_samples, load_relation
from src.models import ModelandTokenizer
from src.utils import experiment_utils, logging_utils
from src.utils.dataclasses import (
    ExperimentResults,
    LayerResult,
    SampleResult,
    TrialResult,
)

logger = logging.getLogger(__name__)


def check_faithfulness(relation: TemporalRelation, lre: JacobianIclMeanEstimator, k=1):
    correct = 0
    wrong = 0

    sample_results: list[SampleResult] = []

    for idx in range(len(relation)):
        sample = relation.samples[idx]
        predictions = lre(sample=sample, k=max(k, 3)).predictions
        known_flag = functional.any_is_nontrivial_prefix(
            predictions=[p.token for p in predictions[:k]], target=sample.object
        )
        logger.debug(
            f'{sample.subject=} ({sample.placeholders["<var>"]}), {sample.object=}, predicted="{functional.format_whitespace(predictions[0].token)}", (p={predictions[0].prob:.3f}), known=({functional.get_tick_marker(known_flag)})',
        )
        correct += known_flag
        wrong += not known_flag

        placeholder_values = list(sample.placeholders.values())
        placeholder_values = ", ".join(placeholder_values)

        sample_results.append(
            SampleResult(
                query=f"{sample.subject} [{placeholder_values}]",
                answer=sample.object,
                prediction=predictions,
            )
        )

    score = correct / (correct + wrong)

    return LayerResult(
        samples=sample_results,
        score=score,
    )


def run_trial(
    mt: ModelandTokenizer,
    relation_file: str,
    layers: list[int],
    relation_size: int,
    num_icl: int,
    prompt_idx: int,
):
    relation = load_relation(
        relation_file=relation_file,
        num_icl=num_icl,
        batch_size=relation_size,
        prompt_idx=prompt_idx,
    )

    # --------------------------------------------------------------------
    # relation = functional.filter_samples_by_var(relation=relation)
    # --------------------------------------------------------------------

    logger.info(
        f"loaded relation {relation.relation_name} | range_stats: {relation.range_stats}"
    )

    relation = functional.filter_samples_by_model_knowledge(
        mt=mt,
        relation=relation,
    )
    relation.samples = balance_samples(relation.samples)

    logger.info(
        f"After filtering by LM knowledge and balancing: {relation.range_stats}"
    )

    trial_result = TrialResult(
        few_shot_demonstration=relation.few_shot_demonstrations,
        faithfulness={},
    )

    for layer in layers:
        estimator = JacobianIclMeanEstimator(
            mt=mt,
            h_layer=layer,
            # -------------------------------------------------------------------
            beta=5.0  # TODO: Eventually, we may want to sweep over this as well - arnab
            # --------------------------------------------------------------------
        )

        lre = estimator(relation)
        faithfulness = check_faithfulness(relation=relation, lre=lre, k=1)
        logger.debug("-" * 80)
        logger.info(
            f"Layer: {layer} | Faithfulness (@1) = {faithfulness.score}, checked with {len(relation.samples)} samples"
        )
        logger.debug("-" * 80)

        trial_result.faithfulness[layer] = faithfulness

    return trial_result


def sweep(
    model_path: str,
    relation_name: str,
    layers: list[int],
    relation_size: int,
    num_icl: int,
    prompt_idx: int,
    num_trials: int,
    results_dir: str,
):
    mt = ModelandTokenizer(model_path=model_path)
    relation_file = relation_name.lower().replace(" ", "_") + ".json"

    results_dir = os.path.join(results_dir, model_path.split("/")[-1])
    results_dir = os.path.join(results_dir, relation_name.lower().replace(" ", "_"))
    results_dir = os.path.join(results_dir, f"{prompt_idx}")

    experiment_results = ExperimentResults(
        experiment_specific_args=dict(
            model=model_path.split("/")[-1],
            relation=relation_name,
            relation_size=relation_size,
            num_icl=num_icl,
        ),
        trial_results=[],
    )

    for trial in range(num_trials):
        logger.debug("-" * 80)
        logger.info(f"Trial {trial+1}/{num_trials}")
        logger.debug("-" * 80)

        trial_result = run_trial(
            mt=mt,
            relation_file=relation_file,
            layers=layers,
            relation_size=relation_size,
            num_icl=num_icl,
            prompt_idx=prompt_idx,
        )
        experiment_results.trial_results.append(trial_result)

        experiment_utils.save_results_file(
            results_dir=results_dir,
            name=f"layer_scores",
            results=experiment_results,
        )

        logger.debug("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sweep over hyperparameters")
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
    )

    parser.add_argument(
        "--relation",
        type=str,
        default="gender head of govt",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(32)),
        help="layers to patch",
    )

    parser.add_argument(
        "--relation-size",
        type=int,
        default=500,  # -1 => load all available samples
        help="number of samples per trial",
    )

    parser.add_argument(
        "--num-icl",
        type=int,
        default=5,
        help="number of few-shot examples to contextualize",
    )

    parser.add_argument(
        "--prompt-idx",
        type=int,
        default=0,
        help="Which prompt template to use (0 => `... <var> ... <subj> ...`, 1 => `... <subj> ... <var> ...`)",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="number of trials per relation",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment = experiment_utils.setup_experiment(args)

    logger.info(args)

    sweep(
        model_path=args.model_path,
        relation_name=args.relation,
        layers=args.layers,
        relation_size=args.relation_size,
        num_icl=args.num_icl,
        prompt_idx=args.prompt_idx,
        num_trials=args.n_trials,
        results_dir=experiment.results_dir,
    )
