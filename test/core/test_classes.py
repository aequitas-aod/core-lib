import unittest
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression

from test import (
    generate_binary_label_dataframe,
    generate_skewed_binary_label_dataframe,
    generate_binary_label_dataframe_with_scores,
    generate_skewed_binary_label_dataframe_with_scores,
    generate_multi_label_dataframe,
    generate_skewed_multi_label_dataframe,
    generate_multi_label_dataframe_with_scores,
    generate_skewed_multi_label_dataframe_with_scores
)

from aequitas.core.datasets import create_dataset

from aequitas.core.imputation import MCMCImputationStrategy
from aequitas.core.metrics import BinaryLabelDatasetScoresMetric

from aequitas.core.algorithms import create_algorithm
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

from aequitas.core.algorithms.preprocessing.optim_preproc_helpers import load_preproc_data_adult_aeq
from aequitas.core.datasets.zoo import adult

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class AbstractMetricTestCase(unittest.TestCase):
    def assertInRange(self, value, lower, upper):
        self.assertGreaterEqual(value, lower)
        self.assertLessEqual(value, upper)

    def assertEDF(self, dataset, dataset_skewed, epsilon=2.337):
        score = dataset.metrics.smoothed_empirical_differential_fairness()
        self.assertFalse(math.isnan(score), msg="EDF should not be nan")
        self.assertIsNotNone(score, msg="EDF should not be None")
        self.assertInRange(score, lower=np.exp(-epsilon), upper=np.exp(epsilon))

        score = dataset_skewed.metrics.smoothed_empirical_differential_fairness()
        self.assertFalse(math.isnan(score), msg="EDF should not be nan")
        self.assertIsNotNone(score, msg="EDF should not be None")
        self.assertFalse(score < np.exp(-epsilon) or score > np.exp(epsilon))

    def assertSP(self, dataset, dataset_skewed, bound=0.1):
        score = dataset_skewed.metrics.statistical_parity_difference()
        self.assertFalse(math.isnan(score), msg="SP should not be nan")
        self.assertIsNotNone(score, msg="SP should not be None")
        self.assertTrue(score > bound or score < -bound, msg="score should not be close to 0 for a biased dataset")

        score = dataset.metrics.statistical_parity_difference()
        self.assertFalse(math.isnan(score), msg="SP should not be nan")
        self.assertIsNotNone(score, msg="SP should not be None")
        self.assertInRange(score, -bound, bound)

    def assertDI(self, dataset, dataset_skewed, bound=0.8):
        score = dataset.metrics.disparate_impact()
        self.assertFalse(math.isnan(score), msg="DI should not be nan")
        self.assertIsNotNone(score, msg="DI should not be None")
        self.assertGreaterEqual(score, bound, f"DI should be >= {bound} for a nonbiased dataset")

        score = dataset_skewed.metrics.disparate_impact()
        self.assertFalse(math.isnan(score), msg="DI should not be nan")
        self.assertIsNotNone(score, msg="DI should not be None")
        self.assertLess(score, bound, msg=f"DI is expected to be below {bound} for biased dataset")

    def assertConsistency(self, dataset, dataset_skewed, bound=0.8):
        score = dataset.metrics.disparate_impact()
        self.assertFalse(math.isnan(score), msg="Consistency should not be nan")
        self.assertIsNotNone(score, msg="Consistency should not be None")
        self.assertGreaterEqual(score, bound, f"Consistency should be >= {bound} for a nonbiased dataset")

        score = dataset_skewed.metrics.disparate_impact()
        self.assertFalse(math.isnan(score), msg="Consistency should not be nan")
        self.assertIsNotNone(score, msg="Consistency should not be None")
        self.assertLess(score, bound, msg=f"Consistency is expected to be < {bound} for biased dataset")

    def assertMeanDifference(self, dataset, dataset_skewed, bound=0.1):
        score = dataset.metrics.mean_difference()
        self.assertFalse(math.isnan(score),
                         msg="Difference in mean outcomes between unprivileged and privileged groups should not be nan")
        self.assertIsNotNone(score,
                             msg="Difference in mean outcomes between unprivileged and privileged groups should not be None")
        self.assertInRange(score, lower=-bound, upper=bound)

        score = dataset_skewed.metrics.mean_difference()
        self.assertFalse(math.isnan(score),
                         msg="Difference in mean outcomes between unprivileged and privileged groups should not be nan")
        self.assertIsNotNone(score,
                             msg="Difference in mean outcomes between unprivileged and privileged groups should not be None")
        self.assertLess(score, bound,
                                f"Difference in mean outcomes between unprivileged and privileged should be < {-bound} in biased dataset")


class TestBinaryLabelDataset(AbstractMetricTestCase):

    def test_dataset_creation_via_factory(self):
        ds = create_dataset("binary label",
                            # parameters of aequitas.BinaryLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.BinaryLabelDataset init
                            favorable_label=1,
                            unfavorable_label=0,
                            # parameters of aif360.StructuredDataset init
                            df=generate_binary_label_dataframe(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr']
                            )

        self.assertIsInstance(ds, BinaryLabelDataset)
        self.assertIsNotNone(ds)

        ds_skewed = create_dataset("binary label",
                                   # parameters of aequitas.BinaryLabelDataset init
                                   unprivileged_groups=[{'prot_attr': 0}],
                                   privileged_groups=[{'prot_attr': 1}],
                                   # parameters of aequitas.StructuredDataset init
                                   imputation_strategy=MCMCImputationStrategy(),
                                   # parameters of aif360.BinaryLabelDataset init
                                   favorable_label=1,
                                   unfavorable_label=0,
                                   # parameters of aif360.StructuredDataset init
                                   df=generate_skewed_binary_label_dataframe(),
                                   label_names=['label'],
                                   protected_attribute_names=['prot_attr']
                                   )
        self.assertIsInstance(ds_skewed, BinaryLabelDataset)
        self.assertIsNotNone(ds_skewed)

    def test_dataset_creation_with_scores_via_factory(self):
        ds = create_dataset("binary label",
                            # parameters of aequitas.BinaryLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.BinaryLabelDataset init
                            favorable_label=1,
                            unfavorable_label=0,
                            # parameters of aif360.StructuredDataset init
                            df=generate_binary_label_dataframe_with_scores(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr'],
                            scores_names="score"
                            )

        self.assertIsInstance(ds, BinaryLabelDataset)
        self.assertIsNotNone(ds)

        ds_skewed = create_dataset("binary label",
                                   # parameters of aequitas.BinaryLabelDataset init
                                   unprivileged_groups=[{'prot_attr': 0}],
                                   privileged_groups=[{'prot_attr': 1}],
                                   # parameters of aequitas.StructuredDataset init
                                   imputation_strategy=MCMCImputationStrategy(),
                                   # parameters of aif360.BinaryLabelDataset init
                                   favorable_label=1,
                                   unfavorable_label=0,
                                   # parameters of aif360.StructuredDataset init
                                   df=generate_skewed_binary_label_dataframe_with_scores(),
                                   label_names=['label'],
                                   protected_attribute_names=['prot_attr']
                                   )

        self.assertIsInstance(ds_skewed, BinaryLabelDataset)
        self.assertIsNotNone(ds_skewed)

    def test_metrics_on_dataset(self):
        ds = create_dataset("binary label",
                            # parameters of aequitas.BinaryLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.BinaryLabelDataset init
                            favorable_label=1,
                            unfavorable_label=0,
                            # parameters of aif360.StructuredDataset init
                            df=generate_binary_label_dataframe_with_scores(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr'],
                            scores_names="score"
                            )

        ds_skewed = create_dataset("binary label",
                                   # parameters of aequitas.BinaryLabelDataset init
                                   unprivileged_groups=[{'prot_attr': 0}],
                                   privileged_groups=[{'prot_attr': 1}],
                                   # parameters of aequitas.StructuredDataset init
                                   imputation_strategy=MCMCImputationStrategy(),
                                   # parameters of aif360.BinaryLabelDataset init
                                   favorable_label=1,
                                   unfavorable_label=0,
                                   # parameters of aif360.StructuredDataset init
                                   df=generate_skewed_binary_label_dataframe_with_scores(),
                                   label_names=['label'],
                                   protected_attribute_names=['prot_attr'],
                                   scores_names="score"
                                   )

        self.assertIsInstance(ds.scores_metrics, BinaryLabelDatasetScoresMetric)
        self.assertIsNotNone(ds)

        self.assertIsInstance(ds_skewed.scores_metrics, BinaryLabelDatasetScoresMetric)
        self.assertIsNotNone(ds_skewed)

        ### METRICS USING LABELS ###

        # Disparate Impact
        self.assertDI(dataset=ds, dataset_skewed=ds_skewed)
        self.assertSP(dataset=ds, dataset_skewed=ds_skewed)
        self.assertEDF(dataset=ds, dataset_skewed=ds_skewed)
        self.assertConsistency(dataset=ds, dataset_skewed=ds_skewed)

        ### METRICS USING SCORES ###

        score = ds.scores_metrics.new_fancy_metric()
        self.assertIsNotNone(score)


class TestMulticlassLabelDataset(AbstractMetricTestCase):
    def test_dataset_creation_via_factory(self):
        ds = create_dataset("multi class",
                            # parameters of aequitas.MulticlassLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.MulticlassLabelDataset init
                            favorable_label=[0, 1., 2.],
                            unfavorable_label=[3., 4.],
                            # parameters of aif360.StructuredDataset init
                            df=generate_multi_label_dataframe(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr']
                            )

        self.assertIsInstance(ds, MulticlassLabelDataset)
        self.assertIsNotNone(ds)

        ds_skewed = create_dataset("multi class",
                                   # parameters of aequitas.MulticlassLabelDataset init
                                   unprivileged_groups=[{'prot_attr': 0}],
                                   privileged_groups=[{'prot_attr': 1}],
                                   # parameters of aequitas.StructuredDataset init
                                   imputation_strategy=MCMCImputationStrategy(),
                                   # parameters of aif360.MulticlassLabelDataset init
                                   favorable_label=[0, 1., 2.],
                                   unfavorable_label=[3., 4.],
                                   # parameters of aif360.StructuredDataset init
                                   df=generate_skewed_multi_label_dataframe(),
                                   label_names=['label'],
                                   protected_attribute_names=['prot_attr']
                                   )
        self.assertIsInstance(ds_skewed, MulticlassLabelDataset)
        self.assertIsNotNone(ds_skewed)
        self.assertDI(dataset=ds, dataset_skewed=ds_skewed)
        self.assertSP(dataset=ds, dataset_skewed=ds_skewed)
        self.assertConsistency(dataset=ds, dataset_skewed=ds_skewed)

    def test_dataset_creation_with_scores_via_factory(self):
        ds = create_dataset("multi class",
                            # parameters of aequitas.MulticlassLabelDataset init
                            unprivileged_groups=[{'prot_attr': 0}],
                            privileged_groups=[{'prot_attr': 1}],
                            # parameters of aequitas.StructuredDataset init
                            imputation_strategy=MCMCImputationStrategy(),
                            # parameters of aif360.MulticlassLabelDataset init
                            favorable_label=[0, 1., 2.],
                            unfavorable_label=[3., 4.],
                            # parameters of aif360.StructuredDataset init
                            df=generate_multi_label_dataframe_with_scores(),
                            label_names=['label'],
                            protected_attribute_names=['prot_attr'],
                            scores_names="score"
                            )

        self.assertIsInstance(ds, MulticlassLabelDataset)
        self.assertIsNotNone(ds)

        ds_skewed = create_dataset("multi class",
                                   # parameters of aequitas.MulticlassLabelDataset init
                                   unprivileged_groups=[{'prot_attr': 0}],
                                   privileged_groups=[{'prot_attr': 1}],
                                   # parameters of aequitas.StructuredDataset init
                                   imputation_strategy=MCMCImputationStrategy(),
                                   # parameters of aif360.MulticlassLabelDataset init
                                   favorable_label=[0, 1., 2.],
                                   unfavorable_label=[3., 4.],
                                   # parameters of aif360.StructuredDataset init
                                   df=generate_skewed_multi_label_dataframe_with_scores(),
                                   label_names=['label'],
                                   protected_attribute_names=['prot_attr'],
                                   scores_names="score"
                                   )
        self.assertIsInstance(ds_skewed, MulticlassLabelDataset)
        self.assertIsNotNone(ds_skewed)


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

        di_remover = create_algorithm(algorithm_type="dir", sensitive_attribute=protected)

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
        protected = "sex"
        ds = load_preproc_data_adult_aeq(unprivileged_groups=[{protected: 0}],
                                         privileged_groups=[{protected: 1}],
                                         protected_attributes=[protected])
        rw = create_algorithm("reweighing", unprivileged_groups=ds.unprivileged_groups,
                              privileged_groups=ds.privileged_groups)
        repaired_ds = rw.fit_transform(ds)
        self.assertMeanDifference(dataset=repaired_ds, dataset_skewed=ds)

    def test_adversarial_debiasing_on_adult_dataset(self):
        attr = "sex"
        u = [{attr: 0}]
        p = [{attr: 1}]
        ds = load_preproc_data_adult_aeq(unprivileged_groups=u,
                                         privileged_groups=p)

        train, test = ds.split([0.7], shuffle=True)

        min_max_scaler = MaxAbsScaler()
        train.features = min_max_scaler.fit_transform(train.features)
        test.features = min_max_scaler.transform(test.features)

        sess = tf.Session()
        plain_model = create_algorithm(algorithm_type="adversarial debiasing",
                                       privileged_groups=p,
                                       unprivileged_groups=u,
                                       scope_name='plain_classifier',
                                       debias=False,
                                       sess=sess)

        plain_model.fit(train)

        nondebiasing_train = plain_model.predict(train)
        nondebiasing_test = plain_model.predict(test)

        sess.close()
        tf.reset_default_graph()
        sess = tf.Session()

        debiased_model = create_algorithm(algorithm_type="adversarial debiasing",
                                          privileged_groups=p,
                                          unprivileged_groups=u,
                                          scope_name='debiased_classifier',
                                          debias=True,
                                          sess=sess)
        debiased_model.fit(train)

        debiasing_train = debiased_model.predict(train)
        debiasing_test = debiased_model.predict(test)

        self.assertMeanDifference(dataset=debiasing_train, dataset_skewed=nondebiasing_train)
        self.assertMeanDifference(dataset=debiasing_test, dataset_skewed=nondebiasing_test)


if __name__ == '__main__':
    unittest.main()

# TODO: test ClassificationMetric, test RegressionMetric
