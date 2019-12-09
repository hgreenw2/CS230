#function that takes as input a .csv file name (string) in the working directory 
#and outputs a dictionary with jpeg names for keys and a 1 or 0 for values
#1 corresponds to at least one defect detected, 0 for no defect
#note, this code comes from the multitask data preprocessing

def constructYbin(filename):
    import csv, numpy as np, copy

    Y = {}  #the final dictionary we want- jpg names point to numpy array of 0s and 1s (which defects exist)
    defectvect = np.zeros(4)
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            #split the file name into jpeg name and defect type
            [img_name, defect_type] = row["ImageId_ClassId"].split('_')
            defect_type = int(defect_type)
        
            #get a boolean, 1 if defect exists, 0 if none exist
            defect_exist = int(row["EncodedPixels"] != "")
            #enter the 1 or 0 into the 4 element numpy array
            defectvect[defect_type - 1] = defect_exist
            line_count += 1

            if (defect_type == 4):  #when defect type is 4, we're at the last defect
                if(sum(defectvect)==0):
                    Y[img_name] = 0
                else:
                    Y[img_name] = 1
    #for key in Y.keys():
        #print(f'{key} -> {Y[key]}')
    #print(f'Processed {line_count} lines.')
        
    return(Y)
