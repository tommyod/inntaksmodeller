#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:56:53 2020

@author: tommy
"""
import random
import collections


def validate_problem(students: list, schools: list):
    """Validate the problem."""

    # Verify data types and attributes
    assert isinstance(students, list)
    assert isinstance(schools, list)
    assert all(hasattr(e, "id") for e in students)
    assert all(hasattr(e, "preferences") for e in students)
    assert all(hasattr(s, "id") for s in schools)
    assert all(hasattr(s, "capacity") for s in schools)

    # IDs must be unique
    assert len(set(e.id for e in students)) == len(students)
    assert len(set(s.id for s in schools)) == len(schools)

    # Preference keys must be school IDs
    school_ids = set(s.id for s in schools)
    for student in students:
        assert all(key in school_ids for key in student.preferences.keys())


def generate_problem_with_popularity(num_students, num_schools, capacity_ratio=1.0, num_choices=3, seed=1):
    """Generate a problem with equal school capacities, but different popularities.
    Student grades are uniformly sampled from [1, 6]."""

    assert num_students % num_schools == 0
    capacity = num_students // num_schools

    # Simple data types for students and schools
    Student = collections.namedtuple("Student", ["id", "preferences", "grade_avg"])
    School = collections.namedtuple("School", ["id", "capacity"])

    # Create schools
    schools = [School(id=i, capacity=int(capacity * capacity_ratio)) for i in range(num_schools)]

    # Set random state for reproducibility
    random_state = random.Random(seed)

    # Create students
    students = []
    for student_id in range(num_students):

        # Compute the grade average for this student
        grade_avg = round(1 + random_state.random() * 5, 1)

        # Create preferences by weighted sampling without replacement
        school_ids = list(range(num_schools))  # Initial pool of schools
        preferences = dict()

        # Sample a school
        for pref in range(1, num_choices + 1):
            weights = [1 / (1 + school_id) for school_id in school_ids]
            school_id = random_state.choices(school_ids, weights, k=1)[0]

            # Add school to preferences and remove it from the pool of schools
            preferences[school_id] = pref
            school_ids.remove(school_id)

        # Add the student to the list of students
        students.append(Student(student_id, preferences, grade_avg))

    assert len(students) == num_students
    assert len(schools) == num_schools
    return students, schools


if __name__ == "__main__":

    students, schools = generate_problem_with_popularity(num_students=500, num_schools=5)
