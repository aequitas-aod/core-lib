# AEQUITAS Core Library

## Introduction

The AEQUITAS core library is one of the key components of the framework proposed by the AEQUITAS European project. The framework seeks to address and combat the various forms of bias and unfairness in AI systems by providing a controlled experimentation environment for AI developers.

This environment allows testing, assessing, and refining socio-technical systems (STS) to identify and mitigate their potentially biased behaviour. Such an environment lets users conduct experiments to evaluate their AI models' performance and behavior under different conditions. The end goal of such evaluation is to facilitate the creation of fairness-compliant AI systems. In other words, this environment empowers developers to make informed decisions about how to understand the fairness related limitations of their AI systems and correct them.

The core library is the component which allows users (precisely developers among all the possible users of the framework) to do essentially two things:

- detect bias within AI models through dedicated metrics
- mitigate the bias (if it exists) using the provided techniques

The library wraps the functionalities provided by the AIF360 library developed by IBM (<https://aif360.res.ibm.com>) while still giving developers the possibility to add their own bias detection or mitigation techniques. More details on the library's whole structure and examples on how its functions can be used as part of code will be given in the next sections.

Overall, we stress that even if the core library is a critical component of the framework proposed by AEQUITAS, its other intended usage is as a standalone Python library for working on AI fairness. In this document it will be presented without describing how it ties to all the other pieces of the framework. The focus will strictly be on the functionalities it provides as a off the shelf library.

## How to use

The first step to use the core library is to install it through pip by running the simple command:

```shell
pip install aequitas-core
```

Once all the required packages have been installed you can start exploiting the library's functionalities inside your own code.

##  Examples

### Creating a dataset

```python
ds = create_dataset(
            dataset_type="binary label",
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            imputation_strategy=MeanImputationStrategy(),
            favorable_label=1.,
            unfavorable_label=0.,
            df=df,
            label_names=['label'],
            protected_attribute_names=['prot_attr']
        )
```

The function `create_dataset` allows to instantiate objects of various classes, each representing a specific type of dataset. In this case, the call to `create_dataset` instantiates an object of type `BinaryLabelDataset`. This class represents datasets for classification purposes whose samples are assigned a label from $\{0,1\}$. In the example, the function is called by passing these parameters:

- `dataset_type`: it is a string which specifies the dataset type. The user won't need to remember the actual type of the object returned by the function because it is all handled by the `create_dataset` function interval
- `unprivileged_groups` and `privileged_groups`: these two parameters are used to distinguish between individuals belonging tho the *unprivileged* and *privileged* groups, depending on the value assigned to the protected attribute `prot_attr`
- `imputation_strategy`: the strategy adopted to impute the missing values is specified by the type of the class passed as value for this parameter. In this case, the class `MeanImputationStrategy` indicates that the missing values will be imputed by relying on the mean of the existing ones
- `unfavorable_label` and `favorable_label`: a label given by a system to an individual can be either *unfavorable* or *favorable* depending on the outcome of the decision made by the system (*i.e.* an unfavorable label corresponds to a negative outcome)
- `df`: the pandas dataframe containing the actual data. It needs to have columns named according to the values of the other two parameters, `label_names` and `protected_attribute_names`

The other supported dataset types are the following:

- `MuticlassLabelDataset`: the elements of this class are those datasets whose samples can be assigned non binary labels (*e.g.* labels from the set $\{0,1, 2, 3, 4\}$).
- `RegressionDataset`: it represents datasets for regression tasks. In these dataset samples are not given any label. Instead, the objective is to predict the value of a given target variable.

To instantiate a `MulticlassLabelDataset` one would call the `create_dataset` function as such:

```python
ds = create_dataset(
            "multi class",
            unprivileged_groups=[{'prot_attr': 0}],
            privileged_groups=[{'prot_attr': 1}],
            imputation_strategy=MCMCImputationStrategy(),
            favorable_label=[0, 1., 2.],
            unfavorable_label=[3., 4.],
            df=df,
            label_names=['label'],
            protected_attribute_names=['prot_attr']
        )
```

The only thing to note in this case, is that the `favorable_label` and `unfavorable_label` parameters are assigned lists and not single values as it happened in the previous example.

Finally, to create a dataset for regression tasks, the call to `create_dataset` would be:

```python
ds = create_dataset(
            dataset_type="regression",
            unprivileged_groups=[{'color': 'b'}],
            privileged_groups=[{'color': 'r'}],
            imputation_strategy=MeanImputationStrategy(),
            df=df,
            dep_var_name='score',
            protected_attribute_names=['color'],
            privileged_classes=[['r']]
        )
```

The parameter `dep_var_name` refers to the *dependent* variable, which, in regression tasks, is the variable whose value has to be predicted.

**Note**: this readme, as well as the library itself is still a work in progress.
