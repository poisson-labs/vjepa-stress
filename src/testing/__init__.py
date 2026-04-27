"""Test fakes used by the runner test suite and the ``--fake`` CLI mode."""

from src.testing.fakes import FakeEncoder, FakeVideoDataset, fake_encoder_factory

__all__ = ["FakeEncoder", "FakeVideoDataset", "fake_encoder_factory"]
