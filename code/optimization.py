#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:23:14 2020

@author: tommy
"""

import time
import numpy as np
import numbers
import itertools
import random

from min_cost_flow import solve_MCF
from utils import validate_problem, generate_problem_with_popularity
from paretoset import paretoset, crowding_distance


class PSSA:
    """Population-based Search with Simulated Annealing."""

    def __init__(
        self, students: list, schools: list, cost_match_func, multi_cost_func, max_frontier_size=8, verbose=0, seed=1
    ):
        """
        Solve school assigment problem using Population-based Search with Simulated Annealing.

        Parameters
        ----------
        students : list
            A list of student objects.
        schools : list
            A list of school objects.
        cost_match_func : callable
            A function with mapping: (student, school) -> non-negative integer cost.
        multi_cost_func : callable
            A function with mapping: (x, cost_matrix, grades, capacities) -> tuple of costs.
            The parameter `x` is a binary matrix of size (students, schools) indicating assignments.
        max_frontier_size : int, optional
            Maximum number of solutions to keep in the Pareto frontier. The default is 8.
        verbose : int, optional
            How much information to print. The default is 0.
        seed : int, optional
            Seed for the random number generator. The default is 1.

        """

        # Verify data types and attributes
        assert callable(cost_match_func)
        validate_problem(students, schools)

        # Set instance parameters
        self.students = students
        self.schools = schools
        self.cost_match_func = cost_match_func
        self.multi_cost_func = multi_cost_func
        self.verbose = verbose
        self.random_state = np.random.RandomState(seed)
        self.random_state_py = random.Random(seed)
        self.student_inds = np.arange(len(students))
        self.max_frontier_size = max_frontier_size

        # Matrix of shape (students, schools) with costs
        self.costs = self.get_cost_matrix(students, schools, cost_match_func)

    @staticmethod
    def get_cost_matrix(students, schools, cost_match_func):
        costs = np.zeros(shape=(len(students), len(schools)), dtype=np.int_)
        # Build the cost matrix
        for i, student in enumerate(students):
            for j, school in enumerate(schools):
                this_cost = cost_match_func(student, school)
                if (not isinstance(this_cost, numbers.Integral)) or this_cost < 0:
                    return TypeError(f"The cost function must return non-negative integers.")
                costs[i][j] = this_cost

        return costs

    @staticmethod
    def get_x_matrix(students, schools, solution_dict):
        # Create a boolean matrix of size (students, schools)
        # This is not strictly needed, but will speed up the algorithm significantly
        x = np.zeros(shape=(len(students), len(schools)), dtype=np.bool_)

        # Go through each student/row
        for i, student in enumerate(students):

            # Get the ID of the school the student was assigned to,
            # find the corresponding column in the boolean matrix and assign it
            assigned_school_id = solution_dict[student.id]
            for j, school in enumerate(schools):
                if school.id == assigned_school_id:
                    x[i][j] = 1
                    break

        return x

    @staticmethod
    def get_search_directions(num_objectives):
        """
        Yield boolean mask for search directions, e.g. [[True, False], [...]]

        Parameters
        ----------
        num_objectives : integer
            Number of objectives (dimensionality of objective function value space).

        Yields
        ------
        list
            List of booleans, with True indicating a search in the direction.

        """

        # One search for each direction in the objective function value space
        for i in range(num_objectives):
            mask = [False] * num_objectives
            mask[i] = True
            yield mask

        # And one search for all directions simultaneously
        yield [True] * num_objectives

    def get_school_inds(self, x):
        """Return indices of two distinct schools."""
        num_students, num_schools = x.shape
        return self.random_state.choice(list(range(num_schools)), size=2, replace=False)

    def get_student_inds(self, x, j1, j2):
        """Return indices of two students attending schools with indices j1 and j2."""
        num_students, num_schools = x.shape

        students_in_j1 = self.student_inds[x[:, j1]]
        students_in_j2 = self.student_inds[x[:, j2]]

        (i1,) = self.random_state.choice(students_in_j1, size=1, replace=True)
        (i2,) = self.random_state.choice(students_in_j2, size=1, replace=True)
        return i1, i2

    @staticmethod
    def domination(solution, best_solution, mask):
        """Does `solution` dominate `best_solution` in the directions specified by the mask?"""
        if not any(mask):
            return True

        solution = [s for (s, m) in zip(solution, mask) if m]
        best_solution = [s for (s, m) in zip(best_solution, mask) if m]

        not_worse = all(s <= b for (s, b) in zip(solution, best_solution))
        one_better = any(s < b for (s, b) in zip(solution, best_solution))
        return not_worse and one_better

    @staticmethod
    def temperature(iteration, start_value=1):
        """Tempareture for simulated annealing, given the iteration."""
        return start_value * 0.99 ** iteration

    def matrix_solution_to_dict(self, x):
        """Conver the boolean matrix solution to a dictionary solution."""
        school_inds = np.arange(len(self.schools))

        solution = dict()
        for i, student in enumerate(self.students):

            # Get the boolean mask for this row
            bool_mask = x[i, :]
            assert np.sum(bool_mask) <= 1, "Each student at most one school"

            # If this student was not placed on any school, break out
            if not np.any(bool_mask):
                continue
            j = school_inds[bool_mask][0]
            solution[student.id] = self.schools[j].id

        return solution

    def operator_move_one(self, x, j_to=None, j_from=None, i=None):

        all_is_None = all(arg is None for arg in (j_to, j_from, i))
        any_is_None = any(arg is None for arg in (j_to, j_from, i))
        assert all_is_None or not any_is_None

        if all_is_None:
            num_students, num_schools = x.shape

            # Get the number of free seats per school
            students_per_school = np.sum(x, axis=0)
            free_seats = self.capacities - students_per_school

            if np.sum(free_seats) == 0 or np.sum(students_per_school) == 0:
                return x, (None, None, None)

            school_inds = np.arange(num_schools)

            # Pick a school with a free seat to move to
            (j_to,) = self.random_state.choice(school_inds[free_seats > 0], size=1, replace=True)

            # Pick a school with students to move from
            mask = np.logical_and(students_per_school > 0, school_inds != j_to)
            schools_to_move_from = school_inds[mask]

            (j_from,) = self.random_state.choice(schools_to_move_from, size=1, replace=True)

            # Choose a student to move
            students_in_j_from = np.arange(num_students)[x[:, j_from]]

            (i,) = self.random_state.choice(students_in_j_from, size=1, replace=True)

            assert not x[i, j_to]
            assert x[i, j_from]

        x[i, j_from], x[i, j_to] = x[i, j_to], x[i, j_from]

        return x, (j_to, j_from, i)

    def operator_swap_two(self, x, j1=None, j2=None, i1=None, i2=None):
        """Mutates the argument and returns the involution."""

        all_is_None = all(arg is None for arg in (j1, j2, i1, i2))
        any_is_None = any(arg is None for arg in (j1, j2, i1, i2))
        assert all_is_None or not any_is_None

        if all_is_None:
            j1, j2 = self.get_school_inds(x)
            i1, i2 = self.get_student_inds(x, j1, j2)

        # Swap two students
        x[i1, j1], x[i1, j2] = x[i1, j2], x[i1, j1]
        x[i2, j1], x[i2, j2] = x[i2, j2], x[i2, j1]

        return x, (j1, j2, i1, i2)

    def solve(self):
        """
        Yields
        ------
        frontier : list
            A list of solutions in the frontier.

        """

        # =============================================================================
        # SOLVE THE MIN-COST FLOW PROBLEM FOR THE INITIAL POPULATION
        # =============================================================================
        MCF_solution = solve_MCF(self.students, self.schools, self.cost_match_func, verbose=self.verbose)

        x = self.get_x_matrix(self.students, self.schools, MCF_solution)

        # =============================================================================
        # PREPARE DATA FOR SIMULATED ANNEALING
        # =============================================================================

        # Vectors of capacities and grades for faster lookups
        self.capacities = np.array([s.capacity for s in self.schools])
        grades = np.array([e.grade_avg for e in self.students])

        # Evaluate the min-cost flow solution, get the number of objectives and store it
        x_objectives = self.multi_cost_func(x, self.costs, grades, self.capacities)
        num_objectives = len(x_objectives)
        frontier = [(x, x_objectives)]
        yield [(MCF_solution, x_objectives)]  # Yield the first frontier containing the the initial solution

        # =============================================================================
        # START SIMULATED ANNEALING
        # =============================================================================

        iterations = list(range(1000))
        search_directions = list(self.get_search_directions(num_objectives))

        # Main loop - continues indefinitely
        for iteration in itertools.count(1):
            if self.verbose > 0:
                print(f"Main search iteration {iteration}.")

            # New solutions found in this main iteration
            solutions_found = []

            for k, (x_frontier, x_objectives) in enumerate(frontier, 1):
                if self.verbose > 2:
                    print(f" Solution {k}/{len(frontier)} in the frontier.")

                # For every search direction in the objective function space
                for search_direction in search_directions:

                    # Initial solution and objectives for the search
                    x = x_frontier.copy()
                    x_objectives = x_objectives

                    # Perform simulated annealing
                    for iteration_direction in iterations:

                        # TODO: These operators waste a lot of time changing students that are
                        # not better off than they already were. This code is just a proof of concept.
                        # In production-code, some time should be spent creating smarter operators.
                        # A simple idea that would work well is sampling with higher probability of
                        # sampling swaps that would make two students better off.

                        # Choose an operator. Swapping is typically better, so choose it more often
                        operators = [self.operator_move_one, self.operator_swap_two]
                        operator = self.random_state_py.choices(operators, weights=[1, 9], k=1)[0]

                        # Change the solution, compute the objectives for the new solution
                        x_before = x.copy()

                        # Both operators are involutions, e.g. x = f(f(x)), so we can return
                        # arguments that make the operators involutions. This way we don't have
                        # to copy the matrix `x` all the time
                        x, involution_args = operator(x)

                        # Evaluate the objectives of the new function
                        x_objectives_new = self.multi_cost_func(x, self.costs, grades, self.capacities)

                        # Does the new solution dominate the old one?
                        dominates = self.domination(x_objectives_new, x_objectives, mask=search_direction)

                        # The initial tempareture is a function of the main ieration
                        temp = self.temperature(iteration_direction, start_value=1 / iteration)

                        # If the new solution is better, store it
                        if dominates or (self.random_state.rand() < temp):
                            x_objectives_best = x_objectives_new
                            solutions_found.append((x.copy(), x_objectives_best))

                        # If it's not better, revert back using involution
                        else:
                            x, _ = operator(x, *involution_args)
                            assert np.all(x == x_before)

            # =============================================================================
            # KEEP ONLY SOLUTIONS IN THE FRONTIER (and filter using crowding distance)
            # =============================================================================
            solutions = frontier + solutions_found

            full_array = np.array([obj for (x, obj) in solutions])
            mask = paretoset(full_array, use_numba=False)
            frontier = [s for (s, m) in zip(solutions, mask) if m]

            crowding_distances = crowding_distance(np.array([obj for (x, obj) in frontier]))
            # Indices of 10 largest elements
            k = min(self.max_frontier_size, len(crowding_distances))
            inds = set(np.argpartition(crowding_distances, -k)[-k:])
            frontier = [f for (i, f) in enumerate(frontier) if i in inds]

            if self.verbose > 1:
                print(f" Found {len(solutions)} candidate solutions and {len(crowding_distances)} efficient solutions.")
                print(f" A total of {len(frontier)} solutions were kept (largest crowding distance).")

            frontier_with_dict = [(self.matrix_solution_to_dict(x), obj) for (x, obj) in frontier]
            yield frontier_with_dict


if __name__ == "__main__":

    # =============================================================================
    # EXAMPLE
    # =============================================================================

    students, schools = generate_problem_with_popularity(
        num_students=1000, num_schools=10, capacity_ratio=1.1, num_choices=3, seed=1
    )

    def cost_match(student, school):
        """Determine integer cost of a student-school assignment."""
        # Grades can also be used (from student.grade_avg), but are not used below

        # If the school is in the wishes
        student_pref = student.preferences.get(school.id)
        if student_pref is not None:
            return student_pref - 1  # Map: 1 -> 0, 2 -> 1, 3 -> 2
        else:
            return 3  # Cost of 3 for a school the student did not want

    def multi_cost_func(x, cost_matrix, grades, capacities):
        """Compute a multi-objective cost of an assigment."""

        # Cost of priorities
        num_students, num_schools = x.shape
        cost_priorities = np.sum(x * cost_matrix) / num_students

        # Standard deviation of mean grades in each school
        mean_grades = [grades[np.arange(num_students)[x[:, i]]].mean() for i in range(num_schools)]
        cost_grades = np.std(mean_grades, ddof=0)

        return cost_priorities, cost_grades

    # Create the algorithm object
    algorithm = PSSA(
        students, schools, cost_match_func=cost_match, multi_cost_func=multi_cost_func, max_frontier_size=20, verbose=2
    )

    import matplotlib.pyplot as plt

    # The .solve() method is implemented as a generator, run X iterations:
    for solutions in itertools.islice(algorithm.solve(), 15):

        solutions = sorted(solutions, key=lambda s: s[1][0])
        x = [x_objectives[0] for (x_frontier, x_objectives) in solutions]
        y = [x_objectives[1] for (x_frontier, x_objectives) in solutions]

        plt.scatter(x, y)
        plt.plot(x, y)
        plt.show()
