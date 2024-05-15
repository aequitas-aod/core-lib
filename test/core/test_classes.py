import unittest
import numpy as np

from sklearn.preprocessing import MinMaxScaler
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

from aequitas.core.datasets import (
    create_dataset,
    BinaryLabelDataset,
    MulticlassLabelDataset
)

from aequitas.core.imputation_strategies import MCMCImputationStrategy
from aequitas.core.metrics import BinaryLabelDatasetScoresMetric

from aequitas.core.algorithms import create_algorithm

from aif360.datasets import AdultDataset


class TestBinaryLabelDataset(unittest.TestCase):

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
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")
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
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")
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
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")

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
        score = ds.metrics.disparate_impact()
        print(f"Disparate impact: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics.disparate_impact()
        print(f"Disparate impact, skewed: {score}")
        self.assertIsNotNone(score)

        # Statistical Parity
        score = ds.metrics.statistical_parity_difference()
        print(f"Statistical Parity: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics.statistical_parity_difference()
        print(f"Statistical Parity, skewed: {score}")
        self.assertIsNotNone(score)

        # Dirichlet-smoothed base rates
        score = ds.metrics._smoothed_base_rates(ds.labels)
        print(f"Dirichlet-smoothed base rates: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics._smoothed_base_rates(ds.labels)
        print(f"Dirichlet-smoothed base rates, skewed: {score}")
        self.assertIsNotNone(score)

        # Smoothed EDF
        score = ds.metrics.smoothed_empirical_differential_fairness()
        print(f"Smoothed EDF: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics.smoothed_empirical_differential_fairness()
        print(f"Smoothed EDF, skewed: {score}")
        self.assertIsNotNone(score)

        # Consistency
        score = ds.metrics.consistency()
        print(f"Consistency: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics.consistency()
        print(f"Consistency, skewed: {score}")
        self.assertIsNotNone(score)

        ### METRICS USING SCORES ###

        score = ds.scores_metrics.new_fancy_metric()
        self.assertIsNotNone(score)


class TestMulticlassLabelDataset(unittest.TestCase):
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
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")
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

        # Disparate Impact
        score = ds.metrics.disparate_impact()
        print(f"Disparate impact: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics.disparate_impact()
        print(f"Disparate impact, skewed: {score}")
        self.assertIsNotNone(score)

        # Statistical Parity
        score = ds.metrics.statistical_parity_difference()
        print(f"Statistical Parity: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics.statistical_parity_difference()
        print(f"Statistical Parity, skewed: {score}")
        self.assertIsNotNone(score)

        # Dirichlet-smoothed base rates
        score = ds.metrics._smoothed_base_rates(ds.labels)
        print(f"Dirichlet-smoothed base rates: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics._smoothed_base_rates(ds.labels)
        print(f"Dirichlet-smoothed base rates, skewed: {score}")
        self.assertIsNotNone(score)

        # Smoothed EDF
        score = ds.metrics.smoothed_empirical_differential_fairness()
        print(f"Smoothed EDF: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics.smoothed_empirical_differential_fairness()
        print(f"Smoothed EDF, skewed: {score}")
        self.assertIsNotNone(score)

        # Consistency
        score = ds.metrics.consistency()
        print(f"Consistency: {score}")
        self.assertIsNotNone(score)

        score = ds_skewed.metrics.consistency()
        print(f"Consistency, skewed: {score}")
        self.assertIsNotNone(score)

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
        mro = False
        if mro:
            print(f"{ds.__class__.__mro__} MRO (aequitas): {ds.__class__.__mro__}")
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


class TestMitigationAlgorithms(unittest.TestCase):

    def test_disparate_impact_remover_on_adult_dataset(self):
        protected = "sex"
        ds = create_dataset("adult",
                            # parameters of aif360.datasets.AdultDataset
                            protected_attribute_names=[protected],
                            privileged_classes=[['Male']], categorical_features=[],
                            features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
                            )

        scaler = MinMaxScaler(copy=False)

        test, train = ds.split([16281])

        train.features = scaler.fit_transform(train.features)
        test.features = scaler.fit_transform(test.features)

        index = train.feature_names.index(protected)

        di_remover = create_algorithm(algorithm_type="dir", sensitive_attribute=protected)

        print(f"disparate impact before mitigation: {test.metrics.disparate_impact()}")

        train_repd = di_remover.fit_transform(train)
        test_repd = di_remover.fit_transform(test)

        X_tr = np.delete(train_repd.features, index, axis=1)
        X_te = np.delete(test_repd.features, index, axis=1)
        y_tr = train_repd.labels.ravel()

        lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
        lmod.fit(X_tr, y_tr)

        test_repd_pred = test_repd.copy()
        test_repd_pred.labels = lmod.predict(X_te)

        print(f"disparate impact after mitigation: {test_repd_pred.metrics.disparate_impact()}")

    def test_reweighing_on_adult_dataset(self):
        protected = "sex"
        ds = create_dataset("adult",
                            # parameters of aif360.datasets.AdultDataset
                            protected_attribute_names=[protected],
                            privileged_classes=[['Male']], categorical_features=[],
                            features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
                            )
        print(
            f"Difference in mean outcomes between unprivileged and privileged groups before reweighing: {ds.metrics.mean_difference()}")
        rw = create_algorithm("reweighing", unprivileged_groups=ds.unprivileged_groups,
                              privileged_groups=ds.privileged_groups)
        repaired_ds = rw.fit_transform(ds)
        print(
            f"Difference in mean outcomes between unprivileged and privileged groups after reweighing: {repaired_ds.metrics.mean_difference()}")


if __name__ == '__main__':
    unittest.main()
