"""
pseudo Code:
read training data
    - split into training set and validation set
    - possibly try k-fold cross validation
parse data
    - all strings must become ints
    - create an array of vectors
    - create an array of corresponding truth values
for validation and prediction sets:
    cosine similarity
    find valid neighbours
    categorization with n nearest neighbours using mean average of neighbours
    ** may need work ** if no valid neighbours use average grade (between G1 and G2 if both exist or whichever grade exists)
    find root mean squared error
    create a submission csv file with predictions

variables to optimize with validation set:
- k nearest neighbours
- threshold for cosine similiarity
- default prediction if no nearest neighbours
"""

f1 = open("data/train_data.csv")
first = True
j = False #delete this, just to test with the first element

for line in f1:
    if first != True and j == False:
        line = line.strip().split(",")
        for indx, attr in enumerate(line):
            print(indx , attr)
        print(line)
        j = True
    else:
        first = False
