myList = ['b', 'r', 'i', 'a', 'n']

for item in myList:
    print ('item')

grid = [['*','*','*','*','*','*'], ['*','.','*','*','.','*',], ['*','*','_','_','*','*',], ['*','*','*','*','*','*',]]

for column in grid:
    row = ""
    for r in column:
        row += r
    print(row)