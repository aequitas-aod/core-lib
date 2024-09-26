import unittest
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression

from test.core import AbstractMetricTestCase
from aequitas.core.imputation import *
from aequitas.core.algorithms.preprocessing import *
from aequitas.core.algorithms.postprocessing import *
from aequitas.core.algorithms.inprocessing import *
from aequitas.core.datasets.zoo import *

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class TestMitigationAlgorithms(AbstractMetricTestCase):

    def test_disparate_impact_remover_on_adult_dataset(self):
        protected = "sex"

        ds = adult(unprivileged_groups=[{protected: 0}],
                   privileged_groups=[{protected: 1}],
                   protected_attribute_names=[protected],
                   privileged_classes=[['Male']],
                   categorical_features=[],
                   features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

        scaler = MinMaxScaler(copy=False)

        test, train = ds.split([16281])

        train.features = scaler.fit_transform(train.features)
        test.features = scaler.fit_transform(test.features)

        index = train.feature_names.index(protected)

        # create_algorithm(algorithm_type="dir", sensitive_attribute=protected)
        di_remover = DisparateImpactRemover(sensitive_attribute=protected)

        train_repd = di_remover.fit_transform(train)
        test_repd = di_remover.fit_transform(test)

        X_tr = np.delete(train_repd.features, index, axis=1)
        X_te = np.delete(test_repd.features, index, axis=1)
        y_tr = train_repd.labels.ravel()

        lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
        lmod.fit(X_tr, y_tr)

        test_repd_pred = test_repd.copy()
        test_repd_pred.labels = lmod.predict(X_te)

        self.assertDI(dataset=test_repd_pred, dataset_skewed=test)

    def test_reweighing_on_adult_dataset(self):
        ds = adult()
        rw = Reweighing(unprivileged_groups=ds.unprivileged_groups, privileged_groups=ds.privileged_groups)
        repaired_ds = rw.fit_transform(ds)
        self.assertMeanDifference(dataset=repaired_ds, dataset_skewed=ds)

    def test_adversarial_debiasing_on_adult_dataset(self):
        ds = adult()
        train, test = ds.split([0.7], shuffle=True)

        min_max_scaler = MaxAbsScaler()
        train.features = min_max_scaler.fit_transform(train.features)
        test.features = min_max_scaler.transform(test.features)

        sess = tf.Session()
        plain_model = AdversarialDebiasing(
            privileged_groups=ds.privileged_groups,
            unprivileged_groups=ds.unprivileged_groups,
            scope_name='plain_classifier',
            debias=False,
            sess=sess
        )

        plain_model.fit(train)

        nondebiasing_train = plain_model.predict(train)
        nondebiasing_test = plain_model.predict(test)

        sess.close()
        tf.reset_default_graph()
        sess = tf.Session()

        debiased_model = AdversarialDebiasing(
            privileged_groups=ds.privileged_groups,
            unprivileged_groups=ds.unprivileged_groups,
            scope_name='debiased_classifier',
            debias=True,
            sess=sess
        )
        debiased_model.fit(train)

        debiasing_train = debiased_model.predict(train)
        debiasing_test = debiased_model.predict(test)

        self.assertMeanDifference(dataset=debiasing_train, dataset_skewed=nondebiasing_train)
        self.assertMeanDifference(dataset=debiasing_test, dataset_skewed=nondebiasing_test)

    def test_deterministic_reranking(self):
        balls = pd.DataFrame(
            [['r', 100], ['r', 90], ['r', 85], ['r', 70], ['b', 70], ['b', 60], ['b', 50], ['b', 40], ['b', 30],
             ['r', 20]],
            columns=['color', 'score'])

        balls_ds = create_dataset(
            dataset_type="regression",
            unprivileged_groups=[{'color': 'b'}],
            privileged_groups=[{'color': 'r'}],
            # parameters of aequitas.StructuredDataset init
            imputation_strategy=DoNothingImputationStrategy(),
            # parameters of aif360.RegressionDataset init
            df=balls,
            dep_var_name='score')

        balls_ds.labels = np.transpose([balls['score']])

        dr = DeterministicReranking(unprivileged_groups=[{'color': 0}], privileged_groups=[{'color': 1}])

        fair_ranking = dr.fit_predict(dataset=balls_ds, rec_size=6, target_prop=[0.5, 0.5], rerank_type='Constrained')
        fair_ranking.convert_to_dataframe()[0]


if __name__ == '__main__':
    unittest.main()

# TODO: test ClassificationMetric, test RegressionMetric
