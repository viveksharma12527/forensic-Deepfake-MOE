class TopKScheduler:
    """Naive example of scheduling of the number of experts (k) selected by noisy top-k gating."""

    def __init__(self, num_epochs: int):
        self.num_epochs = num_epochs

    def __call__(self, epoch: int) -> int:
        # TODO: with the current gating network implementation (model/gating_networks.py: line 76) do not return k=1.
        #  This is a known issue: see https://arxiv.org/pdf/1701.06538, pag. 4, Section "Training the Gating Network".
        #  If k=1 is required at training time, see model/gating_networks.py: line 96 and change the routing mechanism.
        return 2


class TemperatureScheduler:
    """Naive example of scheduling the gating softmax temperature."""

    def __init__(self, min_temperature: float = 0.05, max_temperature: float = 1.0):
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    def __call__(self, epoch: int) -> float:
        return 1.0
