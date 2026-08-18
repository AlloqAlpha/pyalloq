from .base import BaseFeatureTransformer
from pyalloq_core.data import MarketData


class FeaturePipeline:
    def __init__(self):
        self.steps = []

    def add(self, transformer: BaseFeatureTransformer) -> "FeaturePipeline":
        self.steps.append(transformer)
        return self

    def run(self, data: MarketData) -> MarketData:
        current_data = data
        for step in self.steps:
            current_data = step.transform(current_data)

        return current_data
