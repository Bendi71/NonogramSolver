from copy import deepcopy
from itertools import combinations_with_replacement
import numpy as np


class NonogramSolver:
    def __init__(self, generated_row_clues, generated_col_clues):
        """
        Initialize the NonogramSolver with row and column clues.

        :param generated_row_clues: List of lists containing row clues.
        :param generated_col_clues: List of lists containing column clues.
        """
        self.row_clues = generated_row_clues
        self.col_clues = generated_col_clues

        self.m = len(generated_row_clues)
        self.n = len(generated_col_clues)

        self.nonogram = np.zeros((self.m, self.n), dtype=int)
        self.row_solutions = []
        self.col_solutions = []

        self.last_nonogram = None

        self.before_col_solutions = None
        self.before_row_solutions = None
        self.before_guess = None

    @staticmethod
    def generate_solutions(clue, length):
        """
        Generate all possible solutions for a given clue and line length.

        :param clue: List of integers representing the clue.
        :param length: Integer representing the length of the line.
        :return: List of lists containing all possible solutions.
        """
        min_length = sum(clue) + len(clue) - 1
        extra_spaces = length - min_length
        space_combinations = combinations_with_replacement(range(len(clue) + 1), extra_spaces)

        solutions = []
        for combination in space_combinations:
            line = [-1] * length
            pos = 0

            for index, count in enumerate(clue):
                if index in combination:
                    pos += combination.count(index)

                for _ in range(count):
                    line[pos] = 1
                    pos += 1

                if pos < length:
                    line[pos] = -1
                    pos += 1

            line[pos:] = [-1] * (length - pos)
            solutions.append(line)
        return solutions

    def get_all_solutions(self):
        """
        Generate all possible solutions for each row and column based on the clues.
        """
        for clues in self.row_clues:
            self.row_solutions.append(self.generate_solutions(clues, self.m))
        for clues in self.col_clues:
            self.col_solutions.append(self.generate_solutions(clues, self.n))

    @staticmethod
    def find_overlapping_cells(solutions):
        """
        Find overlapping cells in a list of solutions.

        :param solutions: List of lists containing possible solutions.
        :return: Tuple (overlapping_black_index, overlapping_whites_index)
        """
        solutions = np.array(solutions)

        overlapping_black_index = []
        overlapping_whites_index = []

        overlap_black = np.all(solutions == 1, axis=0)
        overlap_whites = np.all(solutions == -1, axis=0)

        overlapping_black_index.extend(np.where(overlap_black)[0])
        overlapping_whites_index.extend(np.where(overlap_whites)[0])

        return overlapping_black_index, overlapping_whites_index

    def update_line_with_overlapping(self, line_index, overlapping_cells, row=True):
        """
        Update a line in the Nonogram grid with overlapping cells.

        :param line_index: Integer representing the index of the line.
        :param overlapping_cells: Tuple (black_index, white_index) containing overlapping cell indices.
        :param row: Boolean indicating whether the line is a row (True) or column (False).
        """
        black_index, white_index = overlapping_cells
        for i in black_index:
            if row:
                self.nonogram[line_index, i] = 1
            else:
                self.nonogram[i, line_index] = 1
        for i in white_index:
            if row:
                self.nonogram[line_index, i] = -1
            else:
                self.nonogram[i, line_index] = -1

    def remove_invalid_solutions(self, line_index, row=True):
        """
        Remove invalid solutions for a given line based on the current Nonogram grid.

        :param line_index: Integer representing the index of the line.
        :param row: Boolean indicating whether the line is a row (True) or column (False).
        """
        if row:
            row_solutions = self.row_solutions[line_index]
            valid_solutions = []

            for solution in row_solutions:
                valid = True
                for j, value in enumerate(solution):
                    if self.nonogram[line_index, j] != 0 and self.nonogram[line_index, j] != value:
                        valid = False
                        break
                if valid:
                    valid_solutions.append(solution)

            self.row_solutions[line_index] = valid_solutions
        else:
            col_solutions = self.col_solutions[line_index]
            valid_solutions = []

            for solution in col_solutions:
                valid = True
                for i, value in enumerate(solution):
                    if self.nonogram[i, line_index] != 0 and self.nonogram[i, line_index] != value:
                        valid = False
                        break
                if valid:
                    valid_solutions.append(solution)

            self.col_solutions[line_index] = valid_solutions

    def update_grid(self):
        """
        Update the Nonogram grid with overlapping cells and remove invalid solutions.
        """
        for i, row in enumerate(self.row_solutions):
            overlapping_cells = self.find_overlapping_cells(row)
            self.update_line_with_overlapping(i, overlapping_cells)
            self.remove_invalid_solutions(i, row=True)
        for i, col in enumerate(self.col_solutions):
            overlapping_cells = self.find_overlapping_cells(col)
            self.update_line_with_overlapping(i, overlapping_cells, row=False)
            self.remove_invalid_solutions(i, row=False)

    def solve_nonogram(self):
        """
        Solve the Nonogram using an iterative approach and guessing if necessary.
        """
        iter_counter = 0
        self.get_all_solutions()
        while not np.all(self.nonogram != 0):
            self.update_grid()
            iter_counter += 1
            if np.array_equal(self.last_nonogram, self.nonogram):
                print("There is no exact solution")
                print("The algorithm can not solve the nonogram. Will continue with guess to find a solution.")
                guess_made = False
                self.save_before_guess()
                for i in range(self.m):
                    for j in range(self.n):
                        if self.nonogram[i, j] == 0:
                            self.nonogram[i, j] = 1
                            self.update_grid()
                            if not np.array_equal(self.last_nonogram, self.nonogram):
                                guess_made = True
                                break
                            else:
                                self.reset_before_guess()
                                self.nonogram[i, j] = -1
                                self.update_grid()
                                if not np.array_equal(self.last_nonogram, self.nonogram):
                                    guess_made = True
                                    break
                            self.reset_before_guess()
                    if guess_made:
                        break
            print(f"Iteration {iter_counter} completed")
            self.last_nonogram = np.copy(self.nonogram)
        if np.all(self.nonogram != 0):
            print(f"Nonogram solved in {iter_counter} iteration!")
        else:
            print(f"Nonogram could not be solved! Iteration stopped at {iter_counter} iteration!")

    def save_before_guess(self):
        """
        Save the current state of the Nonogram grid and solutions before making a guess.
        """
        self.before_guess = deepcopy(self.nonogram)
        self.before_row_solutions = deepcopy(self.row_solutions)
        self.before_col_solutions = deepcopy(self.col_solutions)

    def reset_before_guess(self):
        """
        Reset the Nonogram grid and solutions to the state before making a guess.
        """
        self.nonogram = self.before_guess
        self.row_solutions = self.before_row_solutions
        self.col_solutions = self.before_col_solutions