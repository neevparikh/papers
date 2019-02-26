This is a file which gets parsed into the readme. <NN> removes a newline.
Everything before and including the line with:
START is removed.
# papers
These are papers which I think are worth reading. No particular order. My interests focus on
hyperparameter/architecture optimization, semi-supervised learning, and computer vision.
NF is not found, WBR is will be released. Importance is just a vague estimate of the importance to me. 
TODO means I haven't read in sufficient detail yet. I haven't always looked very hard for all the 
associated links and I might have messed up some of them. PRs with more papers or fixs are welcome. 

| name | arXiv | pdf | github | reddit | open review | desc | misc | importance |
|---|---|---|---|---|---|---|---|---|
| AdaBound | [arxiv](https://arxiv.org/abs/1607.01097) | [pdf](https://arxiv.org/pdf/1607.01097.pdf) | <NN>
[github](https://github.com/Luolc/AdaBound) | <NN>
[reddit](https://old.reddit.com/r/MachineLearning/comments/auvj3q/r_adabound_an_optimizer_that_<NN>
trains_as_fast_as/) | [open review](https://openreview.net/forum?id=Bkg3g2R9FX) | optimizer | new, <NN>
not proven | 5/10 |
| Evaluating the Search Phase of NAS | [arxiv](https://arxiv.org/abs/1902.08142) | <NN>
[pdf](https://arxiv.org/pdf/1902.08142) | WBR |<NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/atwnlh/r_evaluating_the_search_phase_of_neur<NN>
al/) | NF | evaluating NAS approaches vs random search | TODO | 8/10 |
| Random Search and Reproducibility for NAS | [arxiv](https://arxiv.org/abs/1902.07638) | <NN>
[pdf](https://arxiv.org/pdf/1902.07638) | [github](https://github.com/liamcli/randomNAS_release) |<NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/atebq8/r_random_search_and_<NN>
reproducibility_for_neural/) | NF | evaluating NAS approaches vs random search | TODO | 8/10 |
| Neural Architecture Optimization | [arxiv](https://arxiv.org/abs/1808.07233) | <NN>
[pdf](https://arxiv.org/pdf/1808.07233.pdf) | [github](https://github.com/renqianluo/NAO) |<NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/9butdc/r_neural_architecture_optimization/) | <NN>
NF | Uses encoder decoder set up for gradient based NAS |  | 4/10 |
| DARTS: Differentiable Architecture Search | [arxiv](https://arxiv.org/abs/1806.09055) | <NN>
[pdf](https://arxiv.org/pdf/1806.09055) | [original](https://github.com/quark0/darts) <NN>
[alternative](https://github.com/khanrc/pt.darts) |<NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/8tzzf0/r_darts_differentiable_architecture_<NN>
search/) | [open review](https://openreview.net/forum?id=S1eYHoC5FX) | Uses scalar multiples of branches <NN>
for continuous relaxation | Alternate implemention is cleaner, but less verified | 8/10 |
| Efficient Neural Architecture Search via Parameter Sharing | [arxiv](https://arxiv.org/abs/1802.03268) <NN>
| [pdf](https://arxiv.org/pdf/1802.03268) | [github](https://github.com/melodyguan/enas) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/7wxdbw/r_efficient_neural_architecture_<NN>
search_via/) | [open review](https://openreview.net/forum?id=ByQZjx-0-) | Like original NAS/PNAS, but <NN>
with weight sharing | implemention is in TF | 5/10 |
| Learning Transferable Architectures for Scalable Image Recognition | <NN>
[arxiv](https://arxiv.org/abs/1707.07012) <NN>
| [pdf](https://arxiv.org/pdf/1707.07012.pdf) | [just the model](https://github.com/wandering007/nasnet<NN>
-pytorch) | [reddit](https://www.reddit.com/r/MachineLearning/comments/6pcurc/r_learning_transferable_<NN>
architectures_for/) | NF | <NN>
 original NASNet | [medium](https://towardsdatascience.com/everything-you-need-to-know-about-automl-and<NN>
-neural-architecture-search-8db1863682bf) | 7/10 |
| ProxylessNAS | [arxiv](https://arxiv.org/abs/1812.00332) | [pdf](https://arxiv.org/pdf/1812.00332) | <NN>
WBR [just the model](https://github.com/MIT-HAN-LAB/ProxylessNAS) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/a3a1xy/r_proxylessnas_direct_neural_<NN>
architecture_search/) | [open review](https://openreview.net/forum?id=HylVB3AqYm) | <NN>
architecture search which adjusts branch probabilities and optimizes for latency | | 8/10 |
