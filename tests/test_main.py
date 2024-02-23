import unittest

import src.src.modules.neat as neat
from main import fitness_function


class TestFitnessFunction(unittest.TestCase):
    def test_fitness_calculation(self):
        # Create a sample genome
        genome = Genome()
        genome_id = 1

        # Create a sample config
        config = Config()

        # Call the fitness_function
        fitness_function([(genome_id, genome)], config)

        # Assert that the fitness value is updated
        self.assertNotEqual(genome.fitness, 0.0)

if __name__ == '__main__':
    unittest.main()