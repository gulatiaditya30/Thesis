import csv

if __name__ == "__main__":

    with open('../images/labeledPlate.csv')as csv_file:
        csvRead = csv.reader(csv_file,delimiter = ',')
        rowNo = 0
        zeros = 0
        ones = 0
        x = list(csvRead)
        print(x[279][9])
        for row in range(0,280,1):
            for col in range(1,10,1):
                if x[row][col] == '0':
                    zeros +=1
                elif x[row][col] == '1':
                    ones += 1
            
            rowNo +=1


        print("zero = "+str(zeros))
        print("ones = "+str(ones))