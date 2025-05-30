import numpy
data = [
    0, 
    1, 1, 1, 
    2, 2, 2, 2, 
    3, 3, 3, 
    4, 4, 
    5, 5, 5, 5, 5, 5, 5, 5, 5, 
    6, 6, 6, 6, 6, 
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
    8, 8, 8, 8, 8, 8, 8, 8, 
    9, 9, 9, 9, 9, 9, 
    10, 10, 10
]

currency = ""
data = sorted(data)

print("MEDIAN")
for i in data:
    print(currency + str(i), end=", ")
print("", end="\n")
print(len(data))
print(currency+ str(numpy.median(data)) + "\n")

print("MEAN")
for i in data:
    print(currency + str(i), end=" + ")
print("", end="\n")
print(currency + str(sum(data)))
print("________________")
print(len(data))
print(currency + str(numpy.mean(data)))
print("", end="\n")

print("MODE")
print()