import aequitas
from aequitas.core import *
import numpy as np
from aequitas.core.conditions import Condition, ConditionLike

Probability = float


def __ensure_finite_ratio(x: Scalar, y: Scalar) -> float:
    if any(is_zero(z) for z in (x, y)):
        return 0.0
    return min(x / y, y / x)


def probability(x: np.array, x_cond: ConditionLike) -> Probability:
    return x_cond(x).mean()


def conditional_probability(y: np.array, y_cond: ConditionLike, x: np.array, x_cond: ConditionLike) -> Probability:
    """Computes the probability of y given x"""
    y_cond = Condition.ensure(y_cond)
    x_cond = Condition.ensure(x_cond)
    x_is_x_value = x_cond(x)
    return y_cond(y[x_is_x_value]).sum() / x_is_x_value.sum()


def discrete_demographic_parities(x: np.array, y: np.array, y_cond: ConditionLike) -> np.array:
    """Computes demographic parity of `x`, w.r.t. `y_cond == True`, assuming that `x` is a discrete variable.

    More formally:
    :math:`dp_i = \|P[f(Y) \mid X = x_i] - P[f(Y)]\|`

    Also see:
        * https://www.ijcai.org/proceedings/2020/0315.pdf, sec. 3, definition 1
        * https://developers.google.com/machine-learning/glossary/fairness?hl=en#demographic-parity

    :param x: (formally :math:`X`) vector of protected attribute (where each component gets values from a **discrete
        distribution**, whose admissible values are :math:`{x_1, x_2, ..., x_n}`)

    :param y: (formally :math:`Y`) vector of predicted outcomes

    :param y_cond: (formally :math:`f`) boolean condition on :math:`Y` w.r.t. which compute demographic parity is
        computed. In case a scalar :math:`y_0` is passed, it is interpreted as the condition :math:`Y = y_0`

    :return: the array :math:`[dp_1, \ldots, dp_n]` (one value for each possible value of `X`)
    """
    y_cond = Condition.ensure(y_cond)
    x_values = np.unique(x)
    prob_y = probability(y, y_cond)
    probabilities = []
    for x_value in (x_values if len(x_values) > 2 else x_values[:1]):
        prob_y_cond = conditional_probability(y, y_cond, x, x_value)
        probabilities.append(abs(prob_y_cond - prob_y))
    return np.array(probabilities)


def __compute_false_rates(x: np.array, y: np.array, y_pred: np.array, x_cond: ConditionLike,
                          y_cond: ConditionLike) -> Probability:
    #  used to compute the differences contained in the array returned by the
    # function discrete_equalised_odds (see its documentation)
    x_cond = Condition.ensure(x_cond)
    x_is_x_value = x_cond(x)
    y_cond = Condition.ensure(y_cond)
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
        distribution**, whose admissible values are :math:`{x_1, x_2, ..., x_n}`)

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
            differences_x.append(__compute_false_rates(x, y, y_pred, x_value, y_value))
        differences.append(differences_x)

    differences = np.array(differences)
    return differences


def discrete_disparate_impact(x: np.array, y: np.array, x_cond: ConditionLike, y_cond: ConditionLike) -> float:
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
        distribution**, whose admissible values are :math:`{0, 1}`)

    :param y: (formally :math:`Y`) vector of values predicted by the binary classifier
    
    :param x_cond: current value assigned to :math:`X`

    :param y_cond: current value assigned to :math:`Y`

    :return: it returns the minimum between the two previously described fractions
    """
    x_cond = Condition.ensure(x_cond)
    y_cond = Condition.ensure(y_cond)

    prob1 = conditional_probability(y, y_cond, x, x_cond)
    prob2 = conditional_probability(y, y_cond, x, x_cond.negate())

    if prob1 == 0.0 or prob2 == 0.0:
        return 0.0
    else:
        return min((prob1 / prob2, prob2 / prob1))


def discrete_equal_opportunity(x: np.array, y: np.array, y_pred: np.array, y_cond: ConditionLike) -> np.array:
    """
        Computes the  equal opportunity for a given classifier h (represented by its predictions h(X)).
        A classifier satisfies the equal opportunity metric if both protected and unprotected groups have equal False
        Negative Rates (FNR), i.e. the probability of a subject in the positive class to have a negative predicted value.

        Formally, in a binary classification task such that Y = 1 (Y = 0) indicates that a certain subject belongs to
        the positive (negative) class and A is the binary sensitive attribute which allows to distinguish between
        protected and unprotected groups, equal opportunity is satisfied by classifier h(X) if:

        :math:`P(h(X) = 0 | Y = 1, A = 1) - P(h(X) = 0 | Y = 1, A = 0)`

        Also see:
            * https://dl.acm.org/doi/10.1145/3194770.3194776, sec. 3.1, definition 3.2.3

        :param x: (formally :math:`X`) vector of protected attribute (where each component gets values from a **discrete
            distribution**, whose admissible values are :math:`{0, 1}`)

        :param y: (formally :math:`Y`) vector of values predicted by the binary classifier

        :param x_cond: current value assigned to :math:`X`

        :param y_cond: current value assigned to :math:`Y`

        :return: it returns a one dimensional array with as many elements as the number of distinct values for the
            protected attribute. For each of these values the equal opportunity metric is computed using the formula
            above.

        """
    x_values = np.unique(x)
    differences = []
    y_cond = Condition.ensure(y_cond)

    for val in x_values:
        differences.append(__compute_false_rates(x=x, y=y, y_pred=y_pred, x_cond=val, y_cond=y_cond))

    return np.array(differences)


def discrete_predictive_parity(x: np.array, y: np.array, y_pred: np.array,
                               y_cond: ConditionLike) -> np.array:
    """
        Computes the predictive parity for a given classifier h (represented by its predictions h(X)).
        A classifier satisfies the predictive parity metric if both protected and unprotected groups have equal PPV
        (Positive Predicted Value). That is, the probability of a subject with positive predcitive value to truly belong
        to the positive class.

        Formally, in a binary classification task such that Y = 1 (Y = 0) indicates that a certain subject belongs to
        the positive (negative) class and A is the binary sensitive attribute which allows to distinguish between
        protected and unprotected groups, predictive parity is satisfied by classifier h(X) if:

        :math:`P(h(X) = 1 | Y = 1, A = 1) - P(h(X) = 1 | Y = 1, A = 0)`

        Also see:
            * https://dl.acm.org/doi/10.1145/3194770.3194776, sec. 3.2, definition 3.2.1

        :param x: (formally :math:`X`) vector of protected attribute (where each component gets values from a **discrete
            distribution**, whose admissible values are :math:`{0, 1}`)

        :param y: (formally :math:`Y`) vector of ground truth values

        :param y_pred (formally :math:`h(X)`): vector of predicted values

        :param y_cond: current value assigned to :math:`Y`

        :return: it returns a one dimensional array with as many elements as the number of distinct values for the
            protected attribute. For each of these values the predictive parity metric is computed using the formula
            above.

    """
    y_cond = Condition.ensure(y_cond)
    y_pred_is_y_value = y_cond(y_pred)

    x_values = np.unique(x)
    probabilities = []
    for x_value in (x_values if len(x_values) > 2 else x_values[:1]):
        x_cond = Condition.ensure(x_value)
        x_is_x_value = x_cond(x)
        x_is_not_x_value = np.bitwise_not(x_is_x_value)

        num1 = y_cond(y[y_pred_is_y_value & x_is_x_value]).sum()
        den1 = (x_is_x_value & y_pred_is_y_value).sum()

        num2 = y_cond(y[y_pred_is_y_value & x_is_not_x_value]).sum()
        den2 = (x_is_not_x_value & y_pred_is_y_value).sum()

        if num1 == 0 or den1 == 0:
            prob1 = 0
        else:
            prob1 = num1 / den1

        if num2 == 0 or den2 == 0:
            prob2 = 0
        else:
            prob2 = num2 / den2

        probabilities.append(abs(prob1 - prob2))
    return np.array(probabilities)


def __compute_bins(pred_probs: np.array):
    scores = np.arange(0.0, 1.1, 0.1)
    bins = [(round(score, 1), 0) for score in scores]

    for bin_ in bins:
        bin_[1] = (np.round(pred_probs, 1) == bin_[0]).sum()

    return bins


def discrete_calibration(x: np.array, y: np.array,
                         pred_probs: np.array,
                         y_cond: ConditionLike) -> np.array:
    scores = np.round(np.arange(0.0, 1.1, 0.1),1)
    
    y_cond = Condition.ensure(y_cond)
    x_values = np.unique(x)

    probabilities = []
    for x_value in x_values:
        x_cond = Condition.ensure(x_value)
        x_is_x_value = x_cond(x)
        row = []
        for score in scores:
            score = Condition.ensure(score)
            pred_prob_is_score = score(np.round(pred_probs, 1))
            num = y_cond(y[pred_prob_is_score & x_is_x_value]).sum()
            den = (pred_prob_is_score & x_is_x_value).sum()
            if num == 0 or den == 0:
                row.append(0.0)
            else:
                row.append(num / den)
        probabilities.append(row)
    return np.array(probabilities)


aequitas.logger.debug("Module %s correctly loaded", __name__)
