import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def single_thread():
    # Tiny matrices otherwise spend most CPU time in BLAS thread scheduling.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
