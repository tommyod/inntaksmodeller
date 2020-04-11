#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:20:09 2020

@author: tommy
"""

import collections
import time

from min_cost_flow import solve_MCF
from utils import validate_problem


def solve_SD(students: list, schools: list, cost_match_func, verbose=0):
    """Solve school allocation using serial dictatorship.
    
    Min-cost flow is used to alloate students to schools when grades are
    identical, instead of using randomness as a tie-breaker."""

    # Verify data types and attributes
    assert callable(cost_match_func)
    validate_problem(students, schools)

    # =============================================================================
    # STEP 1: SET UP DATA STRUCTURES
    # =============================================================================

    # Group the students by grades. This is needed to ensure fairness for
    # students with identical grade averages
    grade_groups = collections.defaultdict(list)
    for student in students:
        grade_groups[student.grade_avg].append(student)

    # Set up a dictionary with capacities
    capacities = {s.id: s.capacity for s in schools}
    School = collections.namedtuple("School", ["id", "capacity"])
    solution = dict()

    # =============================================================================
    # STEP 2: SOLVE THE PROBLEM
    # =============================================================================
    start_time = time.time()

    # Loop over groups of students with identical grades in decreasing order
    for grade, students_group in sorted(grade_groups.items(), reverse=True):
        if verbose > 0:
            print(f"Allocating {len(students_group)} students with grade {grade}.")

        # Pseudo-schools with updated capacities used to solve the sub-problem
        # Data is copied so as not to mutate the input arguments
        pseudo_schools = [School(id=s.id, capacity=capacities[s.id]) for s in schools]

        # Solve the problem for this group of students using min-cost flow
        sub_solution = solve_MCF(students_group, pseudo_schools, cost_match_func, verbose=verbose)

        # Students assigned in this round should never be in the solution already
        assert set(solution.keys()).intersection(set(sub_solution.keys())) == set()

        # Update the full solution with the solution for this group of students
        solution.update(sub_solution)

        # Update school capacities by decrementing
        for school_id in sub_solution.values():
            capacities[school_id] -= 1

        # Capacities should never be non-negative
        assert all(c >= 0 for c in capacities.values())

    solve_time = time.time() - start_time
    if verbose > 0:
        print(f"Solved serial dictatorship in {solve_time} seconds.")

    return solution


if __name__ == "__main__":

    # Simple data types for students and schools
    Student = collections.namedtuple("Student", ["id", "preferences", "grade_avg"])
    School = collections.namedtuple("School", ["id", "capacity"])

    # Problem data: schools
    schools = [School(1, 1), School(2, 1), School(3, 1), School(4, 1)]

    # Problem data: students
    students = [
        Student("Ola", {1: 1, 4: 2}, 5.0),  # The preference map is 'school.id' -> 'wish number'
        Student("Elev2", {1: 1, 2: 2}, 4.9),
        Student("Elev3", {2: 1, 3: 2}, 4.9),
        Student("Elev4", {3: 1, 4: 2}, 4.9),
    ]

    def cost_match(student, school):
        """Determine integer cost of a student-school assignment."""
        # Grades can also be used, but are not used here

        # If the school is in the wishes
        student_pref = student.preferences.get(school.id)
        if student_pref is not None:
            return student_pref - 1
        else:
            return 2

    solution = solve_SD(students, schools, cost_match_func=cost_match, verbose=1)
