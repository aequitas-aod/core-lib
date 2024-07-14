# AEQUITAS Core Library

The AEQUITAS core library is one of the key components of the framework proposed by the AEQUITAS European project. The framework seeks to address and combat the various forms of bias and unfairness in AI systems by providing a controlled experimentation environment for AI developers.

This environment allows testing, assessing, and refining socio-technical systems (STS) to identify and mitigate their potentially biased behaviour. Such an environment lets users conduct experiments to evaluate their AI models' performance and behavior under different conditions. The end goal of such evaluation is to facilitate the creation of fairness-compliant AI systems. In other words, this environment empowers developers to make informed decisions about how to understand the fairness related limitations of their AI systems and correct them.

The core library is the component which allows users (precisely developers among all the possible users of the framework) to do essentially two things:

- detect bias within AI models through dedicated metrics
- mitigate the bias (if it exists) using the provided techniques

The library wraps the functionalities provided by the AIF360 library developed by IBM (https://aif360.res.ibm.com) while it also gives developers the possibility to add their own bias detection or bias mitigation techniques. More details on the library's whole structure and examples on how its functions can be used as part of code will be given in the next sections. 

Overall, we stress that even if the core library is a critical component of the framework proposed by AEQUITAS, its other intended usage is as a standalone Python library for working on AI fairness.
