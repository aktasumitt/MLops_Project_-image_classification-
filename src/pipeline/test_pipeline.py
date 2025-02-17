from src.components.testing.testing import Testing
from src.config.configuration import Configuration

class TestPipeline():
    def __init__(self):
        self.config=Configuration()

    def run_test(self):
        test_config=self.config.get_testing_configs()
        testing=Testing(test_config)
        test_result = testing.initiate_testing()

        return test_result
    
if __name__=="__main__":

    testpipeline=TestPipeline()
    testpipeline.run_test()
