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
[reddit](https://reddit.com/r/MachineLearning/comments/auvj3q/r_adabound_an_optimizer_that_<NN>
trains_as_fast_as/) | [open review](https://openreview.net/forum?id=Bkg3g2R9FX) | optimizer | new, <NN>
not proven, TODO | 6/10 |
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
NF | Uses encoder decoder for gradient based NAS |  | 4/10 |
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
| Deep Residual Learning for Image Recognition | [arxiv](https://arxiv.org/abs/1512.03385) <NN>
| [pdf](https://arxiv.org/pdf/1512.03385) | [github](https://github.com/KaimingHe/deep-residual-networks)<NN>
| [reddit](https://www.reddit.com/r/MachineLearning/comments/3wb6p9/msras_deep_residual_learning_for_<NN>
image_recognition/) | NF | ResNets | <NN>
[medium](https://medium.com/datadriveninvestor/paper-summary-deep-residual-learning-for-image-<NN>
recognition-8c5ecedc6478) | 10/10 |
| FractalNet | [arxiv](https://arxiv.org/abs/1605.07648) <NN>
| [pdf](https://arxiv.org/pdf/1605.07648) | [github](https://github.com/khanrc/pt.fractalnet)<NN>
| [reddit](https://www.reddit.com/r/MachineLearning/comments/4lexdd/fractalnet_ultradeep_neural_<NN>
networks_without/) | [open review](https://openreview.net/forum?id=S1VaB4cex) | <NN>
fractal architecture to reduce path length | residual nets won the war | 5/10 |
| Identity Mappings in Deep Residual Networks | [arxiv](https://arxiv.org/abs/1603.05027) <NN>
| [pdf](https://arxiv.org/pdf/1603.05027) | [github](https://github.com/KaimingHe/resnet-1k-layers)<NN>
| [reddit](https://www.reddit.com/r/MachineLearning/comments/4asemg/160305027_identity_mappings_in_<NN>
deep_residual/) | NF | ResNet v2 | | 8/10 |
| Wide Residual Networks | [arxiv](https://arxiv.org/abs/1605.07146) <NN>
| [pdf](https://arxiv.org/pdf/1605.07146) | [github](https://github.com/szagoruyko/wide-residual-<NN>
networks) | [reddit](https://www.reddit.com/r/MachineLearning/comments/4krp3w/<NN>
wide_residual_networks/) | NF | ResNet variant with wider layers and less depth | | 7/10 |
| EraseReLU | [arxiv](https://arxiv.org/abs/1709.07634) <NN>
| [pdf](https://arxiv.org/pdf/1709.07634) | NF | [reddit](https://www.reddit.com/r/MachineLearning/<NN>
comments/72e4ju/r_eraserelu_a_simple_way_to_ease_the_training_of/) | NF | ResNet variant with <NN>
removed ReLU | | 2/10 |
| Deep Networks with Stochastic Depth | [arxiv](https://arxiv.org/abs/1603.09382) <NN>
| [pdf](https://arxiv.org/pdf/1603.09382) | [github](https://github.com/yueatsprograms/Stochastic_Depth)<NN>
| [reddit](https://www.reddit.com/r/MachineLearning/comments/4curk5/160309382v1_deep_networks_with_<NN>
stochastic_depth/) | [open review](https://openreview.net/forum?id=BJm63h2F) | introduces stochastic <NN>
depth in resnets | | 8/10 |
| Deep Pyramidal Residual Networks | [arxiv](https://arxiv.org/abs/1610.02915) <NN>
| [pdf](https://arxiv.org/pdf/1610.02915v1.pdf) | [github](https://github.com/dyhan0920/PyramidNet-<NN>
PyTorch) | [reddit](https://www.reddit.com/r/MachineLearning/comments/5rr84i/r_arxiv161002915_deep_<NN>
pyramidal_residual_networks/) | NF | optimizes resnet architecture including additive channels | <NN>
[medium](https://medium.com/@sh.tsang/review-pyramidnet-deep-pyramidal-residual-networks-image-<NN>
classification-85a87b60ae78)  best simple architecture for classification IMO | 9/10 |
| Deep Pyramidal Residual Networks with Separated Stochastic Depth | <NN>
[arxiv](https://arxiv.org/abs/1612.01230) | [pdf](https://arxiv.org/pdf/1612.01230) | <NN>
[github](https://github.com/AkTgWrNsKnKPP/PyramidNet_with_Stochastic_Depth) | NF | <NN>
[open review](https://openreview.net/forum?id=SkPxL0Vte) | tests ResDrop in pyramid net | <NN>
not great paper or very innovative IMO, including for relevance to ShakeDrop | 1/10 |
| Shake-Shake regularization | [arxiv](https://arxiv.org/abs/1705.07485) <NN>
| [pdf](https://arxiv.org/pdf/1705.07485) | [github](https://github.com/xgastaldi/shake-shake) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/5vo14r/r_shakeshake_regularization_of_<NN>
3branch_residual/) | [open review](https://openreview.net/forum?id=HkO-PCmYl) | stochasticly combines <NN>
output from multiple branches | | 6/10 |
| ShakeDrop Regularization for Deep Residual Learning | [arxiv](https://arxiv.org/abs/1802.02375) <NN>
| [pdf](https://arxiv.org/pdf/1802.02375) | [github](https://github.com/imenurok/ShakeDrop) | NF | <NN>
[open review](https://openreview.net/forum?id=S1NHaMW0b) | regularization technique for pyramid net | <NN>
under rated IMO| 6/10 |
|  MobileNets | [arxiv](https://arxiv.org/abs/1704.04861) <NN>
| [pdf](https://arxiv.org/pdf/1704.04861) | [github](https://github.com/tensorflow/models/blob/master<NN>
/research/slim/nets/mobilenet_v1.md) | [reddit](https://www.reddit.com/r/MachineLearning/comments<NN>
/663m43/r_170404861_mobilenets_efficient_convolutional/) | NF | lightweight architecture for mobile<NN>
, depthwise separable convolution | implemention is in TF <NN>
[blog](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html)| 6/10 |
|  MobileNet v2 | [arxiv](https://arxiv.org/abs/1801.04381) | [pdf](https://arxiv.org/pdf/1801.04381) | <NN>
[github](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/89g2b1/r_mobilenetv2<NN>
_the_next_generation_of_ondevice/) | NF | improved mobilenet | implemention is in TF <NN>
[blog](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)| 6/10 |
| ShuffleNet | [arxiv](https://arxiv.org/abs/1707.01083) <NN>
| [pdf](https://arxiv.org/pdf/1707.01083) | [github](https://github.com/jaxony/ShuffleNet) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/6lj295/r_170701083_shufflenet_<NN>
an_extremely_efficient/) | NF | lightweight architecture for mobile, introduces channel shuffle || 7/10 |
| ShuffleNet v2 | [arxiv](https://arxiv.org/abs/1807.11164) <NN>
| [pdf](https://arxiv.org/pdf/1807.11164) | [github](https://github.com/ericsun99/Shufflenet-v2-Pytorch)<NN>
| NF | NF | specific hardware focused optimizations | | 5/10 |
| Super-Convergence | [arxiv](https://arxiv.org/abs/1708.07120) <NN>
| [pdf](https://arxiv.org/pdf/1708.07120) | [github](https://github.com/lnsmith54/super-convergence)<NN>
| [reddit](https://www.reddit.com/r/MachineLearning/comments/6vnp6b/r_170807120_superconvergence_<NN>
very_fast_training/) | [open review](https://openreview.net/forum?id=H1A5ztj3b) | <NN>
approach for using learning rate schedule to speed up training | in practice leads to higher variance <NN>
in final validation loss, works very well in general | 7/10 |
| Decoupled Weight Decay Regularization | [arxiv](https://arxiv.org/abs/1711.05101) <NN>
| [pdf](https://arxiv.org/pdf/1711.05101) | [pr](https://github.com/pytorch/pytorch/pull/3740)<NN>
| [reddit](https://www.reddit.com/r/MachineLearning/comments/7d5qob/r_171105101_fixing_weight_<NN>
decay_regularization_in/) | [initial open review](https://openreview.net/forum?id=rk6qdGgCZ) <NN>
[second open review](https://openreview.net/forum?id=Bkg6RiCqY7) | adam works better with decoupled <NN>
weight decay | always use decoupled weight decay | 7/10 |
| Parallel Architecture and Hyperparameter Search via Successive Halving and Classification | <NN>
[arxiv](https://arxiv.org/abs/1805.10255) | [pdf](https://arxiv.org/pdf/1805.10255) | <NN>
[github](https://github.com/titu1994/pyshac) | NF | NF | search approach | simple, not best choice for <NN>
architecture search IMO | 5/10 |
| Fixup Initialization | [arxiv](https://arxiv.org/abs/1901.09321) | <NN>
[pdf](https://arxiv.org/pdf/1901.09321) | <NN>
[github](https://github.com/ajbrock/BoilerPlate/blob/master/Models/fixup.py) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/amw8i0/r_fixup_initialization_<NN>
residual_learning_without/) | [open review](https://openreview.net/forum?id=H1gsz30cKX) | <NN>
initialization scheme which allows for no batchnorm | very architecture specific | 4/10 |
| Squeeze-and-Excitation Networks | [arxiv](https://arxiv.org/abs/1709.01507) | <NN>
[pdf](https://arxiv.org/pdf/1709.01507) | <NN>
[github](https://github.com/moskomule/senet.pytorch) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/6yg3ak/r170901507_<NN>
squeezeandexcitation_networks_imagenet/) | NF | novel block for CNN | seems like an easy win | 9/10 |
| Batch Renormalization | [arxiv](https://arxiv.org/abs/1702.03275) | <NN>
[pdf](https://arxiv.org/pdf/1702.03275) | <NN>
[github](https://github.com/titu1994/BatchRenormalization) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/5tr0cd/r_batch_renormalization_<NN>
towards_reducing/) | NF | approach for using batch normalization when the batch size must be small | <NN>
I found that it doesn't make much difference unless the batch size is very small, implemention is in <NN>
keras | 5/10 |
| GroupNorm | [arxiv](https://arxiv.org/abs/1803.08494) | [pdf](https://arxiv.org/pdf/1803.08494) | <NN>
[github](https://github.com/kuangliu/pytorch-groupnorm) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/86gurl/r_group_normalization_fair/) | <NN>
NF | alternative to batch normalization, better when batch must be very small | | 5/10 |
| Attention Is All You Need | [arxiv](https://arxiv.org/abs/1706.03762) | <NN>
[pdf](https://arxiv.org/pdf/1706.03762) | <NN>
[github](https://github.com/jadore801120/attention-is-all-you-need-pytorch) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/6gwqiw/r_170603762_attention_is_all_<NN>
you_need_sota_nmt/) | [NeurIPS Review](https://media.nips.cc/nipsbooks/nipspapers/paper_files/<NN>
nips30/reviews/3058.html) | introduced transformer model for NLP | [medium](https://medium.com/@<NN>
adityathiruvengadam/transformer-architecture-attention-is-all-you-need-aeccd9f50d09) TODO | 9/10 |
| Deep Contextualized Word Representations (ELMo) | [arxiv](https://arxiv.org/abs/1802.05365) | <NN>
[pdf](https://arxiv.org/pdf/1802.05365) | <NN>
[github](https://github.com/allenai/bilm-tf) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/7xwp1t/r_180205365_deep_<NN>
contextualized_word/) | NF | better embeddings | [blog](https://www.mihaileric.com/posts/deep-<NN>
contextualized-word-representations-elmo/) implemention is in TF, TODO | 5/10 |
| BERT | [arxiv](https://arxiv.org/abs/1810.04805) | [pdf](https://arxiv.org/pdf/1810.04805) | <NN>
[github](https://github.com/huggingface/pytorch-pretrained-BERT) | <NN>
[reddit](https://www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_<NN>
of_deep_bidirectional/) | NF | state of the art NLP model which is easy to fine tune for a variety <NN>
of tasks | variety of explanations online, [illustrated BERT](http://jalammar.github.io/<NN>
illustrated-bert/), TODO | 9/10 |



