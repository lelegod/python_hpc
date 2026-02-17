class Student:
    def __init__(self, name, courses):
        self.name = name
        self.courses = courses
    def attends(self, course):
        return True if course in self.courses else False