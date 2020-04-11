#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:18:47 2020

@author: tommy
"""

import collections
import numbers
from ortools.graph import pywrapgraph
import time
from utils import validate_problem


def solve_MCF(students: list, schools: list, cost_match_func, verbose=0):
    """Solve school allocation as a Min Cost Flow problem."""

    # Verify data types and attributes
    assert callable(cost_match_func)
    validate_problem(students, schools)

    # =============================================================================
    # STEP 1: SET UP THE MIN COST FLOW PROBLEM
    # =============================================================================

    start_time = time.time()

    num_schools = len(schools)
    num_students = len(students)

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Names for source and sink nodes
    source = 0
    sink = 1 + num_students + num_schools

    # Add arcs from source (node 0) to students
    node_to_student_id = dict()
    for i, student in enumerate(students):
        min_cost_flow.AddArcWithCapacityAndUnitCost(source, 1 + i, 1, 0)
        node_to_student_id[i + 1] = student.id
        if verbose > 2:
            print(f"Added arc from node {source} to {i+1} with capacity 1.")

    # Add arcs from students to schools
    for i, student in enumerate(students):
        for j, school in enumerate(schools):
            this_cost = cost_match_func(student, school)
            if (not isinstance(this_cost, numbers.Integral)) or this_cost < 0:
                return TypeError(f"The cost function must return non-negative integers.")

            min_cost_flow.AddArcWithCapacityAndUnitCost(1 + i, 1 + num_students + j, 1, this_cost)
            if verbose > 2:
                print(f"Added arc from node {1+i} to {1+num_students+j} with capacity {1} and cost {this_cost}.")

    # Add arcs from schools to target
    node_to_school_id = dict()
    for j, school in enumerate(schools):
        node_to_school_id[1 + num_students + j] = school.id
        min_cost_flow.AddArcWithCapacityAndUnitCost(1 + num_students + j, sink, school.capacity, 0)

        if verbose > 2:
            print(f"Added arc from node {1+num_students+j} to {sink} with capacity {school.capacity}.")

    # Set supply and demand
    min_cost_flow.SetNodeSupply(source, num_students)
    min_cost_flow.SetNodeSupply(sink, -num_students)
    solve_time = time.time() - start_time
    if verbose > 0:
        print(f"Set up min-cost flow problem in {solve_time} seconds.")

    # =============================================================================
    # STEP 2: SOLVE THE MIN COST FLOW PROBLEM
    # =============================================================================

    start_time = time.time()
    # Same as Solve(), but does not have the restriction that the supply must match
    # the demand or that the graph has enough capacity to serve all the demand or
    # use all the supply. This will compute a maximum-flow with minimum cost.
    result = min_cost_flow.SolveMaxFlowWithMinCost()
    solve_time = time.time() - start_time
    assert result == min_cost_flow.OPTIMAL
    if verbose > 0:
        print(f"Solved min-cost flow problem in {solve_time} seconds.")

    # =============================================================================
    # STEP 3: PARSE THE RESULTS OF THE SOLUTION AND RETURN
    # =============================================================================

    start_time = time.time()

    chosen_schools = dict()
    for arc in range(min_cost_flow.NumArcs()):

        # We ignore arcs leading out of source or into sink.
        if min_cost_flow.Tail(arc) == source:
            continue
        if min_cost_flow.Head(arc) == sink:
            continue

        assert min_cost_flow.Flow(arc) in (0, 1)
        if min_cost_flow.Flow(arc) < 1:
            continue

        # Add solution the dictionary
        student_id = node_to_student_id[min_cost_flow.Tail(arc)]
        school_id = node_to_school_id[min_cost_flow.Head(arc)]
        chosen_schools[student_id] = school_id

    solve_time = time.time() - start_time
    if verbose > 0:
        print(f"Parsed min-cost flow solution in {solve_time} seconds.")

    return chosen_schools


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

    solution = solve_MCF(students, schools, cost_match_func=cost_match, verbose=1)

    for student_id, school_id in solution.items():
        print(f"Student '{student_id}' was assigned to school '{school_id}'.")
