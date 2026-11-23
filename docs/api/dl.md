# Deep Learning API Reference

This page auto-generates documentation from `pyalloq.dl` docstrings.

---

## Dataset, Costs & Walk-Forward Training

::: pyalloq.dl.dataset.MarketDataset

::: pyalloq.dl.dataset.UniverseSubsampler

::: pyalloq.dl.costs.TorchCostAdapter

::: pyalloq.dl.trainer.WalkForwardTrainer

---

## Set Transformer Backbone & Specialized Layers

::: pyalloq.dl.layers.set_transformer.ScalableMultiheadAttention

::: pyalloq.dl.layers.set_transformer.ISAB

::: pyalloq.dl.layers.set_transformer.PMA

::: pyalloq.dl.layers.set_transformer.SetTransformerBackbone

::: pyalloq.dl.layers.relational_graph.RelationalGraphPriorModule

::: pyalloq.dl.layers.neural_jump_rde.TorchLogSignatureExtractor

::: pyalloq.dl.layers.neural_jump_rde.NeuralJumpRDE

::: pyalloq.dl.layers.mdfp.MDFPOptimizationLayer

::: pyalloq.dl.layers.fragility_kkt.FragilityShrinkageCholeskyLayer

::: pyalloq.dl.layers.fragility_kkt.DifferentiableRiskParityLayer

::: pyalloq.dl.layers.fragility_kkt.DifferentiableCVaRLayer

::: pyalloq.dl.policy.LogSigSSMBlock

::: pyalloq.dl.policy.UnifiedE2EPolicyNetwork

::: pyalloq.dl.policy.compute_von_neumann_entropy

---

## Deep Return, Covariance & End-to-End Allocators

::: pyalloq.dl.estimators.returns.deep_return.DeepReturnEstimator

::: pyalloq.dl.estimators.returns.models.tft.TFTDeepReturnsNet

::: pyalloq.dl.estimators.covariance.deep_cov.DeepCovarianceEstimator

::: pyalloq.dl.estimators.covariance.models.set_covariance.SetTransformerCovarianceNet

::: pyalloq.dl.estimators.covariance.models.gnn.CRISPSparseSTGAT

::: pyalloq.dl.estimators.covariance.models.cac.CrossAttentionCholeskyNet

::: pyalloq.dl.estimators.e2e.deep_allocator.DeepAllocator

::: pyalloq.dl.estimators.e2e.ipmo_allocator.IPMOAllocator

::: pyalloq.dl.estimators.e2e.models.set_policy.SetTransformerPolicyNet

---

## Foundation Models, Ensembles & Neural SDE

::: pyalloq.dl.estimators.apollopfn.ApolloPFNReturnEstimator

::: pyalloq.dl.estimators.apollopfn.ApolloPFNViewGenerator

::: pyalloq.dl.estimators.apollopfn.RiemannBarDistribution

::: pyalloq.dl.estimators.ensemble.EnsembleReturnEstimator

::: pyalloq.dl.estimators.ensemble.EnsembleViewGenerator

::: pyalloq.dl.estimators.sde_estimator.NeuralSDEEstimator

---

## Self-Supervised World Models (`MarketJEPA`)

::: pyalloq.dl.world_models.jepa.MarketJEPA

::: pyalloq.dl.world_models.jepa.VICRegLoss

::: pyalloq.dl.world_models.jepa.LatentPredictor

---

## Differentiable Loss Functions

::: pyalloq.dl.loss_functions.woodbury.WoodburyGaussianNLLLoss

::: pyalloq.dl.loss_functions.woodbury.WoodburySteinLoss

::: pyalloq.dl.loss_functions.woodbury.SoftMinEVaRLoss

::: pyalloq.dl.loss_functions.cholesky_nll.CholeskyNLLLoss

::: pyalloq.dl.loss_functions.sharpe.SharpeLoss

::: pyalloq.dl.loss_functions.dfl.SPOPlusLoss

::: pyalloq.dl.loss_functions.dfl.HybridDFLLoss
