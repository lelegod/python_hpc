def coursestudents(students, course):
    return [student.name for student in students if course in student.courses]