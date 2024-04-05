import os

def clustalW_MSA(requestsForOneCluster):
    print("\nStarting ClustalW....")
    requestsForOneCluster_postClustalW = []
    # Open the file in write mode
    with open("inputalign.fa", "w") as input_file:
        # Iterate through requestsForOneCluster and write each request to the file
        for request in requestsForOneCluster:
            
            # Strip off curly braces and write the modified request to the file
            modified_request = request.replace("{", "").replace("}", "")
            input_file.write(">text\n")
            input_file.write(modified_request + '\n')  # Write the modified request to the file with a newline character
    
    # Run the mafft command
    os.system("mafft --text inputalign.fa > outputalign.fa")
    
    with open("outputalign.fa", "a") as output_file:
        output_file.write(">text\n")
            
    with open("outputalign.fa", "r") as file:
        lines = file.readlines()
        requestsForOneCluster_postClustalW = []
        # read the file line by line and everything between two ">text" should be counted as one request
        request = ""
        for line in lines:
            #  if it starts with >text or you reached the end of the file
            if line.startswith(">text"):
                if request != "":
                    requestsForOneCluster_postClustalW.append("{" + request + "}")  # Enclose the request within curly braces
                request = ""
            else:
                request += line.strip()
        print("\nClustalW MSA")
        for req in requestsForOneCluster_postClustalW:
            print(req)
        return requestsForOneCluster_postClustalW

