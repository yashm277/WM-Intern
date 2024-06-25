def getAlignedScore(seq1,seq2):
    """
    Needleman Wunsch Algorithm to calculate the similarity score between two sequences.
    The algorithm uses a scoring system for match, mismatch and gap.
    The algorithm uses a dynamic programming approach to calculate the alignment score.
    A higher score means a higher similarity between the two sequences.
    """

    m = len(seq1)                                   #Seq1 will be the vertical sequence to the left
    n = len(seq2)                                   #Seq2 will be the horizontal sequence on top
    init_mat = []                                   #Initialised matrix

    #Scoring system for match, mismatch and gap:
    match = 1
    mismatch = -1
    gap = -1

    #Initialising the matrix to 0 (Part1):
    for i in range(m+1):                            #Adding +1 as one extra column needed to hold the initialised values
        temp = []
        for j in range(n+1):                        #Adding +1 as one extra column needed to hold the initialised values
            temp.append(0)
        init_mat.append(temp)                       #init_mat is a matrix of len(seq1)+1 rows and len(seq2)+1 columns containing 0's

    for j in range(n+1):
        init_mat[0][j] = gap*j                      #Multiplying 0,1,2... with the gap in order to initialise the matrix 1st row

    for i in range(m+1):
        init_mat[i][0] = gap*i                      #Multiplying 0,1,2... with the gap penatly in order to initilaise the matrix 1st column

    #Matrix filling (Part2):
    for i in range(1,m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                init_mat[i][j] = max(init_mat[i][j-1]+gap, init_mat[i-1][j]+gap, init_mat[i-1][j-1]+match)
            else:
                init_mat[i][j] = max(init_mat[i][j-1]+gap, init_mat[i-1][j]+gap, init_mat[i-1][j-1]+mismatch)

    '''
    #Visualise as a matrix
    for row in init_mat:
        for element in row:
            print(element, end="\t")
        print("\n")
    '''

    #Backtracking (Part3):
    seq1_align = ""                                 #The algined sequence (seq1) is going to be appended to this string
    seq2_align = ""                                 #The aligned sequence (seq2) is going to be appended to this string
    score = 0

    i = m                                           #i = m = len(seq1)
    j = n                                           #j = n = len(seq2)

    while (i>0 or j>0):

        #Checking if it is a match. If it is a match, then append and jump to the diagonal value directly:
        if seq1[i-1] == seq2[j-1]:
            seq1_align += seq1[i-1]
            seq2_align += seq2[j-1]
            i -= 1
            j -= 1

        #If the sequence don't match:
        elif seq1[i-1] != seq2[j-1]:
            temp_list = [init_mat[i-1][j-1], init_mat[i-1][j], init_mat[i][j-1]]        #Creating a temp_list in order to find the maximum values from top, diagonal and left in order to backtrack

            #If the maximum value is the 0th indexed position, i.e., the diagonal value:
            if max(temp_list) == temp_list[0]:
                seq1_align += seq1[i-1]
                seq2_align += seq2[j-1]
                i -= 1
                j -= 1

            #If the maximum value is the 1st indexed position, i.e., the top value:
            elif max(temp_list) == temp_list[1]:
                seq1_align += seq1[i-1]
                seq2_align += "-"
                i -= 1

            #If the maximum value is the 2nd indexed position, i.e., the left vlaue:
            elif max(temp_list) == temp_list[-1]:
                seq1_align += "-"
                seq2_align += seq2[j-1]
                j-=1

        #If there is an error (somehow? just in case?), initialising the values of i and j in order for it to not turn into an infinite loop
        else:
            print("Error. Exit.")
            i=0
            j=0

    seq1_align = seq1_align[::-1]                   #Reverse the string seq1_align
    seq2_align = seq2_align[::-1]                   #Reverse the string seq2_align

    #Storing the match, mismatch and gap symbols in match_string:
    match_string = ""
    for i in range(len(seq1_align)):
        if seq1_align[i] == seq2_align[i]:
            match_string += "|"
        elif seq1_align[i] != seq2_align[i]:
            if (seq1_align[i] == "-" or seq2_align[i] == "-"):
                match_string += " "
            else:
                match_string += "*"

    #Calculating the alignment score:
    alignment_score = 0
    for i in range(len(match_string)):
        if match_string[i] == "|":
            alignment_score += match
        elif match_string[i] == "*":
            alignment_score += mismatch
        else: 
            alignment_score+=gap
    return alignment_score