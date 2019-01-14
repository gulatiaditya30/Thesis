import csv

if __name__ == "__main__":

    with open('../images/labeledPlate.csv')as csv_file:
        csvRead = csv.reader(csv_file,delimiter = ',')
        rowNo = 0
        zeros = 0
        ones = 0
        x = list(csvRead)
        print(x[279][0])
        for row in csvRead:
            for val in row:
                if val == '0':
                    zeros +=1
                elif val == '1':
                    ones += 1
            
            rowNo +=1


        print("zero = "+str(zeros))
        print("ones = "+str(ones))