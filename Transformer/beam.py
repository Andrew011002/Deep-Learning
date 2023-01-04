import torch


class Beam:

    def __init__(self, model, beam_size, maxlen, device):
        self.model = model
        self.beam_size = beam_size
        self.maxlen = maxlen
        self.device = device

    def beam_search(self, sequences, start, end):
        pass