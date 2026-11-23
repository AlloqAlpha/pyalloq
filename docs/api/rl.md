# Reinforcement Learning API Reference

This page auto-generates documentation from `pyalloq.rl` docstrings.

---

## Environments

::: pyalloq.rl.envs.portfolio_env.PortfolioGymEnv

::: pyalloq.rl.envs.portfolio_env.ScenarioPortfolioEnv

::: pyalloq.rl.envs.pomdp_env.POMDPPortfolioEnv

::: pyalloq.rl.envs.meta_env.MetaPortfolioEnv

---

## Agents & Pipeline Allocators

::: pyalloq.rl.agents.base.BaseRLAgent

::: pyalloq.rl.agents.sb3_wrapper.SB3PortfolioAgent

::: pyalloq.rl.allocator.RLAllocator

::: pyalloq.rl.agents.meta_allocator.HierarchicalMetaRLAllocator

---

## Reward Functions

::: pyalloq.rl.rewards.functions.BaseRewardFunction

::: pyalloq.rl.rewards.functions.DifferentialSharpeReward

::: pyalloq.rl.rewards.functions.StepwiseSortinoReward

::: pyalloq.rl.rewards.functions.CARAUtilReward

---

## Wrappers

::: pyalloq.rl.wrappers.dict_features.DictFlattenWrapper

::: pyalloq.rl.wrappers.dict_features.FeatureNormalizationWrapper

::: pyalloq.rl.wrappers.endogenous_impact.EndogenousLiquidityImpactWrapper

---

## Continuous-Time Exploratory RL & Regime Gates

::: pyalloq.rl.exploratory_mv.WonhamRegimeFilter

::: pyalloq.rl.exploratory_mv.ExploratoryRegimeMVAllocator

::: pyalloq.rl.thermodynamics.WassersteinRegimeGate

::: pyalloq.rl.thermodynamics.SignatureMMDRegimeGate

---

## Pathwise Deep Hedging

::: pyalloq.rl.trajectory.TrajectoryPolicyNet

::: pyalloq.rl.trajectory.DeepHedgingTrajectoryTrainer
