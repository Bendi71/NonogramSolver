import random
import matplotlib.pyplot as plt
from NonogramSolver.extract import image_to_nonogram
from NonogramSolver.solver import NonogramSolver


class Nonogram:
    def __init__(self, size):
        """
        Initialize a Nonogram object with a given size.

        :param size: Tuple (m, n) representing the dimensions of the Nonogram.
        """
        self.m, self.n = size
        self.nonogram = None
        self.original = None
        self.solver = None

    def generate_random_nonogram(self):
        """
        Generate a random Nonogram grid with 0s and 1s.
        """
        self.nonogram = [[random.choice([0, 1]) for _ in range(self.m)] for _ in range(self.n)]

    def generate_clues(self):
        """
        Generate row and column clues for the current Nonogram grid.

        :return: Tuple (generated_row_clues, generated_col_clues)
        """
        generated_row_clues = [self.extract_clues(row) for row in self.nonogram]
        generated_col_clues = [self.extract_clues(col) for col in zip(*self.nonogram)]
        return generated_row_clues, generated_col_clues

    @staticmethod
    def extract_clues(line):
        """
        Extract clues from a single line (row or column) of the Nonogram grid.

        :param line: List of integers (0s and 1s) representing a line in the Nonogram.
        :return: List of integers representing the clues for the line.
        """
        clues = []
        count = 0
        for cell in line:
            if cell == 1:
                count += 1
            elif count > 0:
                clues.append(count)
                count = 0
        if count > 0:
            clues.append(count)
        return clues or [0]

    def print_nonogram(self):
        """
        Print the current Nonogram grid to the console.
        """
        for row in self.nonogram:
            print('---------------------')
            print(' | ' + ''.join('X' if cell == 1 else ' ' for cell in row) + ' | ')

    def image_to_nonogram(self, image_path, threshold=128):
        """
        Convert an image to a Nonogram grid.

        :param image_path: Path to the image file.
        :param threshold: Threshold value for converting the image to binary.
        """
        nonogram = image_to_nonogram(image_path, size=(self.m, self.n), threshold=threshold)
        self.original = nonogram
        self.nonogram = nonogram

    def print_original(self):
        """
        Display the original Nonogram image using matplotlib.
        """
        plt.imshow(self.original, cmap='binary', interpolation='nearest', aspect='equal')
        plt.show()

    def setup_solver(self):
        """
        Set up the Nonogram solver with the generated clues.
        """
        generated_row_clues, generated_col_clues = self.generate_clues()
        self.solver = NonogramSolver(generated_row_clues, generated_col_clues)

    def solve(self):
        """
        Solve the Nonogram using the solver.
        """
        self.setup_solver()
        self.solver.solve_nonogram()
        self.nonogram = self.solver.nonogram

    def print_image(self):
        """
        Display the solved Nonogram grid using matplotlib.
        """
        plt.imshow(self.nonogram, cmap='binary', interpolation='nearest', aspect='equal')
        plt.show()

    def print_difference(self):
        """
        Display the difference between the original and solved Nonogram grids using matplotlib.
        """
        if self.nonogram is not None and self.original is not None:
            difference = self.nonogram - self.original
            plt.imshow(difference, cmap='binary', interpolation='nearest', aspect='equal')
            plt.show()
        else:
            print("Nonogram or original image is not initialized.")