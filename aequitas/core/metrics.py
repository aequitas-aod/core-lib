import typing

from aequitas import logger
from aequitas.core import *
import eagerpy


EPSILON: float = 1e-9
INFINITY: float = 1e9
DELTA: float = 5e-2  # percentage to apply to the values of the protected attribute to create the buckets


@returning_tensor
def single_conditional_probability(predicted: Tensor,
                                   protected: Tensor,
                                   value: int,
                                   equal: bool = True) -> eagerpy.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param value: the value of the protected attribute.
    @param equal: if True, filter rows whose protected attribute is equal to value,
    otherwise filter rows whose protected attribute is not equal to value.
    @return: the conditional probability.
    """
    predicted = ensure_tensor(predicted)
    protected = ensure_tensor(protected)
    ensure_same_tensor_type(predicted, protected)
    if equal:
        mask = predicted[protected == value]
    else:
        mask = predicted[protected != value]
    return mask.mean()


@returning_tensor
def single_conditional_probability_in_range(predicted: Tensor,
                                            protected: Tensor,
                                            min_value: float,
                                            max_value: float,
                                            inside: bool = True) -> eagerpy.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param min_value: the minimum value of the protected attribute.
    @param max_value: the maximum value of the protected attribute.
    attribute is not equal to value.
    @param inside: if True, filter rows whose protected attribute is inside the range, otherwise filter rows whose
    protected attribute is outside the range.
    @return: the conditional probability.
    """
    predicted = ensure_tensor(predicted)
    protected = ensure_tensor(protected)
    ensure_same_tensor_type(predicted, protected)
    if inside:
        mask = predicted[eagerpy.logical_and(protected >= min_value, protected < max_value)]
    else:
        mask = predicted[eagerpy.logical_or(protected < min_value, protected >= max_value)]
    return mask.mean() if len(mask) > 0 else eagerpy.zeros(type(predicted), (1,))


def demographic_parity(p: Tensor, y: Tensor, epsilon: float = EPSILON, continuous: bool = False,
                       numeric: bool = True, delta: float = DELTA) -> typing.Union[bool, Tensor]:
    """
    Demographic parity is a measure of fairness that measures if a value of a protected feature impacts the outcome of a
    prediction. In other words, it measures if the outcome is independent of the protected feature.
    The protected feature must be binary or categorical.
    The output must be binary.
    :param p: protected feature
    :param y: output
    :param epsilon: threshold for demographic parity
    :param delta: approximation parameter for the calculus of continuous demographic parity
    :param continuous: if True, calculate the continuous demographic parity
    :param numeric: if True, return the value of demographic parity instead of a boolean
    :return: True if demographic parity is less than epsilon, False otherwise
    """
    p = ensure_tensor(p)
    y = ensure_tensor(y)
    ensure_same_tensor_type(p, y)
    absolute_probability = y.mean()
    parity = 0

    def _continuous_demographic_parity() -> float:
        result = 0
        min_protected = p.min()
        max_protected = p.max()
        interval = max_protected - min_protected
        step_width = interval * delta
        number_of_steps = int(interval / step_width)
        for i in range(number_of_steps):
            min_value = min_protected + i * step_width
            max_value = min_protected + (i + 1) * step_width
            cond_probability = single_conditional_probability_in_range(y, p, min_value, max_value)
            if cond_probability == 0:
                continue
            n_samples = eagerpy.logical_and(p >= min_value, p < max_value).sum()
            result += (cond_probability - absolute_probability).abs() * n_samples
        return result / len(p)

    if continuous:
        parity = _continuous_demographic_parity()
    else:
        unique_p = unique(p)
        for p_value in unique_p:
            conditional_probability = single_conditional_probability(y, p, p_value)
            if conditional_probability == 0:
                continue
            number_of_sample = (p == p_value).sum()
            parity += (conditional_probability - absolute_probability).abs() * number_of_sample
        parity /= len(p)
    return parity < epsilon if not numeric else parity


# def disparate_impact(
#         p: np.array, y: np.array, threshold: float = DISPARATE_IMPACT_THRESHOLD, continuous: bool = False,
#         numeric: bool = True, delta: float = DELTA
# ) -> bool:
#     """
#     Disparate impact is a measure of fairness that measures if a protected feature impacts the outcome of a prediction.
#     It has been defined on binary classification problems as the ratio of the probability of a positive outcome given
#     the protected feature to the probability of a positive outcome given the complement of the protected feature.
#     If the ratio is less than a threshold (usually 0.8), then the prediction is considered to be unfair.
#     The protected feature must be binary or categorical.
#     The output must be binary.
#     :param p: protected feature
#     :param y: output
#     :param threshold: threshold for disparate impact
#     :param continuous: if True, calculate the continuous disparate impact
#     :param numeric: if True, return the value of disparate impact instead of a boolean
#     :param delta: approximation parameter for the calculus of continuous disparate impact
#     :return: True if disparate impact is less than threshold, False otherwise
#     """
#     unique_protected = np.unique(p)
#
#     def _continuous_disparate_impact() -> float:
#         result = 0
#         min_protected = np.min(p)
#         max_protected = np.max(p)
#         interval = max_protected - min_protected
#         step_width = interval * delta
#         number_of_steps = int(interval / step_width)
#         for i in range(number_of_steps):
#             min_value = min_protected + i * step_width
#             max_value = min_protected + (i + 1) * step_width
#             conditional_probability_in = single_conditional_probability_in_range(y, p, min_value, max_value)
#             conditional_probability_out = single_conditional_probability_in_range(y, p, min_value, max_value,
#                                                                                   negate=True)
#             if conditional_probability_in <= EPSILON or conditional_probability_out <= EPSILON:
#                 pass
#             else:
#                 number_of_sample = np.sum(np.logical_and(p >= min_value, p < max_value))
#                 ratio = conditional_probability_in / conditional_probability_out
#                 inverse_ratio = conditional_probability_out / conditional_probability_in
#                 result += min(ratio, inverse_ratio) * number_of_sample
#         return result / len(p)
#
#     if continuous:
#         impact = _continuous_disparate_impact()
#     else:
#         probabilities_a = np.array([np.mean(y[p == x]) for x in unique_protected])
#         probabilities_not_a = np.array([np.mean(y[p != x]) for x in unique_protected])
#         first_impact = np.nan_to_num(probabilities_a / probabilities_not_a)
#         second_impact = np.nan_to_num(probabilities_not_a / probabilities_a)
#         number_of_samples = np.array([np.sum(p == x) for x in unique_protected])
#         pair_wise_weighted_min = np.min(np.vstack((first_impact, second_impact)), axis=0) * number_of_samples
#         impact = np.sum(pair_wise_weighted_min) / len(p)
#     fairness.logger.info(f"Disparate impact: {impact:.4f}")
#     return impact > threshold if not numeric else impact
#
#
# def equalized_odds(
#         p: np.array, y_true: np.array, y_pred: np.array, epsilon: float = EPSILON, continuous: bool = False,
#         numeric: bool = True
# ) -> bool:
#     """
#     Equalized odds is a measure of fairness that measures if the output is independent of the protected feature given
#     the label Y.
#     The protected feature must be binary or categorical.
#     The output must be binary.
#     :param p: protected feature
#     :param y_true: ground truth
#     :param y_pred: prediction
#     :param epsilon: threshold for equalized odds
#     :param numeric: if True, return the value of equalized odds instead of a boolean
#     :return: True if equalized odds is satisfied, False otherwise
#     """
#     conditional_prob_zero = np.mean(y_pred[y_true == 0])
#     conditional_prob_one = np.mean(y_pred[y_true == 1])
#     unique_protected = np.unique(p)
#
#     def _continuous_equalized_odds() -> float:
#         min_protected = np.min(p)
#         max_protected = np.max(p)
#         interval = max_protected - min_protected
#         step_width = interval * DELTA
#         number_of_steps = int(interval / step_width)
#         result = 0
#         for i in range(number_of_steps):
#             probs_a_0 = np.array([np.mean(y_pred[(p >= min_protected + i * step_width) & (
#                         p < min_protected + (i + 1) * step_width) & (y_true == 0)])])
#             probs_a_1 = np.array([np.mean(y_pred[(p >= min_protected + i * step_width) & (
#                         p < min_protected + (i + 1) * step_width) & (y_true == 1)])])
#             n_samples = np.array([np.sum(
#                 (p >= min_protected + i * step_width) & (p < min_protected + (i + 1) * step_width) & (y_true == y)) for
#                                           y in [0, 1]])
#             partial = np.abs(
#                 np.concatenate([probs_a_0 - conditional_prob_zero, probs_a_1 - conditional_prob_one]))
#             partial = np.nan_to_num(partial)
#             partial = np.sum(partial * n_samples)
#             result += partial
#         return result / len(y_true)
#
#     if continuous:
#         eo = _continuous_equalized_odds()
#     else:
#         probabilities_a_0 = np.array([np.mean(y_pred[(p == x) & (y_true == 0)]) for x in unique_protected])
#         probabilities_a_1 = np.array([np.mean(y_pred[(p == x) & (y_true == 1)]) for x in unique_protected])
#         number_of_samples = np.array([np.sum((p == x) * (y_true == y)) for x in unique_protected for y in [0, 1]])
#         eo = np.abs(
#             np.concatenate([probabilities_a_0 - conditional_prob_zero, probabilities_a_1 - conditional_prob_one]))
#         eo = np.nan_to_num(eo)
#         eo = np.sum(eo * number_of_samples) / np.sum(number_of_samples)
#     fairness.logger.info(f"Equalized odds: {eo:.4f}")
#     return eo < epsilon if not numeric else eo


# let this be the last line of this file
logger.debug("Module %s correctly loaded", __name__)
