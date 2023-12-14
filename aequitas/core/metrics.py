import aequitas
from aequitas.core import *
import numpy as np
import typing


Probability = float
Condition = typing.Callable[[np.array], np.array]
ConditionOrScalar = typing.Union[Condition, Scalar]


def __ensure_is_condition(condition_or_value: ConditionOrScalar) -> Condition:
    if aequitas.isinstance(condition_or_value, Scalar):
        def check_condition(vector: np.array) -> Condition:
            return vector == condition_or_value
        return check_condition
    else:
        return condition_or_value


def __ensure_finite_ratio(x: Scalar, y: Scalar) -> float:
    if any(is_zero(z) for z in (x, y)):
        return 0.0
    return min(x / y, y / x)


def probability(x: np.array, x_cond: ConditionOrScalar) -> Probability:
    return x_cond(x).mean()


def conditional_probability(
    y: np.array,
    y_cond: ConditionOrScalar,
    x: np.array,
    x_cond: ConditionOrScalar,
) -> Probability:
    """Computes the probability of y given x"""
    y_cond = __ensure_is_condition(y_cond)
    x_cond = __ensure_is_condition(x_cond)
    x_is_x_value = x_cond(x)
    return y_cond(y[x_is_x_value]).sum() / x_is_x_value.sum()


def discrete_demographic_parities(x: np.array, y: np.array, y_cond: ConditionOrScalar) -> np.array:
    """Computes demographic parity of `x`, w.r.t. `y_cond == True`, assuming that `x` is a discrete variable.

    More formally:
    :math:`dp_i = \|P[f(Y) \mid X = x_i] - P[f(Y)]\|`

    Also see:
        * https://www.ijcai.org/proceedings/2020/0315.pdf, sec. 3, definition 1
        * https://developers.google.com/machine-learning/glossary/fairness?hl=en#demographic-parity

    :param x: (formally :math:`X`) vector of protected attribute (where each component gets values from a **discrete
        distribution**, whose admissible values are :math:`{x_1, x_2, ..., x_n}`

    :param y: (formally :math:`Y`) vector of predicted outcomes

    :param y_cond: (formally :math:`f`) boolean condition on :math:`Y` w.r.t. which compute demographic parity is
        computed. In case a scalar :math:`y_0` is passed, it is interpreted as the condition :math:`Y = y_0`

    :return: the array :math:`[dp_1, \ldots, dp_n]` (one value for each possible value of `X`)
    """
    y_cond = __ensure_is_condition(y_cond)
    x_values = np.unique(x)
    prob_y = probability(y, y_cond)
    probabilities = []
    for x_value in (x_values if len(x_values) > 2 else x_values[:1]):
        prob_y_cond = conditional_probability(y, y_cond, x, x_value)
        probabilities.append(abs(prob_y_cond - prob_y))
    return np.array(probabilities)


def __compute_false_rates(x: np.array, y: np.array, y_pred: np.array, x_cond: ConditionOrScalar,
                       y_cond: ConditionOrScalar) -> Probability:
    # used to compute the differences contained in the array returned by the
    # function discrete_equalised_odds (see its documentation)
    x_cond = __ensure_is_condition(x_cond)
    x_is_x_value = x_cond(x)
    y_cond = __ensure_is_condition(y_cond)
    y_is_not_y_value = np.bitwise_not(y_cond(y))

    cond1 = y_cond(y_pred[y_is_not_y_value & x_is_x_value]).sum() / (x_is_x_value & y_cond(y)).sum()
    cond2 = y_cond(y_pred[y_is_not_y_value]).sum() / (y_cond(y)).sum()
    return abs(cond1 - cond2)


def discrete_equalised_odds(x: np.array, y: np.array, y_pred: np.array) -> np.array:
    """Computes the equalised odds for a given classifier h (represented by its predictions h(X)).
        A classifier satisfies equalised odds if its predictions are independent of the protected
        attribute given the labels. The following must hold for all unique values of Y and all the unique values of X. 

    More formally:
        :math:`eo_ij = \|P[h(X) \mid X = x_j, Y = y_i] - P[h(X) \mid Y = y_i]\|`

    Also see:
        * https://www.ijcai.org/proceedings/2020/0315.pdf, sec. 3, definition 2

    :param x: (formally :math:`X`) vector of protected attribute (where each component gets values from a **discrete
        distribution**, whose admissible values are :math:`{x_1, x_2, ..., x_n}`

    :param y: (formally :math:`Y`) vector of ground truth values
    
    :param y_pred: (formally :math:`h(X)`) vector of predicted values

    :return: a math:`m x n` array where :math:`m` is the number of unique values of Y and :math:`n` is the number 
        of unique values of X. Each element of the array :math:`eo` contains the previously defined difference. """
    
    x_values = np.unique(x)
    y_values = np.unique(y)
        
    differences = []

    for y_value in y_values:
        differences_x = []
        for x_value in x_values:
            differences_x.append(__compute_false_rates(x, y, y_pred, x_value,
                                                       y_value))
        differences.append(differences_x)
    
    differences = np.array(differences)
    return differences

def discrete_disparate_impact(x: np.array,
                              y: np.array,
                              x_cond: ConditionOrScalar,
                              y_cond: ConditionOrScalar) -> float:
    """
    Computes the disparate impact for a given classifier h (represented by its predictions h(X)).
    A classifier suffers from disparate impact if its predictions disproportionately hurt people
    with certain sensitive attributes. It is defined as the minimum between two fractions. 

    One fraction is:

    :math:`P(h(X) = 1 | X = 1) / P(h(X) = 1 | X = 0)`

    while the other is its reciprocal. If the minimum between the two is exactly 1 then the classifier
    doesn't suffer from disparate impact.

    Also see:
        * https://www.ijcai.org/proceedings/2020/0315.pdf, sec. 3, definition 3

    :param x: (formally :math:`X`) vector of protected attribute (where each component gets values from a **discrete
        distribution**, whose admissible values are :math:`{0, 1}`

    :param y: (formally :math:`Y`) vector of values predicted by the binary classifier
    
    :param x_cond: current value assigned to :math:`X`

    :param y_cond: current value assigned to :math:`Y`

    :return: it returns the minimum between the two previously described fractions
    """    

    prob1 = conditional_probability(y, y_cond, x, x_cond)
    prob2 = conditional_probability(y, y_cond, x, abs(x_cond - 1))

    if prob1 == 0.0 or prob2 == 0.0:
        return 0.0
    else:
        return min((prob1/prob2, prob2/prob1))



aequitas.logger.debug("Module %s correctly loaded", __name__)
