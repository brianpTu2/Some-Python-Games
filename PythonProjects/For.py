grid = [['S', 'F', 'F', 'F'], ['F', 'F', 'N', 'F'], ['N', 'F', 'F', 'F'], ['N', 'F', 'F', 'G']]
print("")
for column in grid:
    row = ""
    for r in column:
        row += r
    print(row)

