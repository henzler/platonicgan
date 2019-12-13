from scripts.tests.test_render import run_test_render
from scripts.tests.test_dataset import run_test_dataset


def run_tests(trainer):
    print('[INFO] Start running tests')
    run_test_render(trainer)
    run_test_dataset(trainer)

    print('[INFO] Finished running tests')
