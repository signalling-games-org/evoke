import unittest

# Discover and run all tests in the 'tests' directory
loader = unittest.TestLoader()
suite = loader.discover(start_dir='evoke/tests/', pattern='test_*.py')

runner = unittest.TextTestRunner()
runner.run(suite)