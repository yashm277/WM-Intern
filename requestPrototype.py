from collections import Counter
import math
# D. Formulating the Request Consensus Prototype and E. Deriving Entropy-Based Positional Weightings
def FormulatingtheRequestPrototype(lines, threshold,b,c):
    # request prototupe calc
    frequencyArray = {}
    for line in lines:
        for i in range(len(line)):  
            if i not in frequencyArray:
                frequencyArray[i] = []  
            frequencyArray[i].append(line[i])
    # print(frequencyArray)
    ans = []
    entropy = []
    weightage = []
    total = len(lines)
    for i in range(len(lines[0])):
        count = Counter(frequencyArray[i])
        key, value = count.most_common()[0]
        if (value/total)>=threshold and key != "-":
            ans.append(key)
        # # this line is causing the error - can increase from 0.5 to 0.8 for example
        # elif (value/total)>=0.5 and key == "-":
        #     continue
        else:
            ans.append("?")
    # print(ans, len(ans))
    reqPrototype = ''.join(ans)
    # entropy calculation
    entropy = [0] * len(lines[0])  
    for i in range(len(lines[0])):
        count = Counter(frequencyArray[i]).items()
        for key, value in count:
            relativeFreq = (value / total)
            entropy[i] += (-1 * relativeFreq * math.log(relativeFreq))
    weightage = [0] * len(lines[0])
    # weightage 
    for i in range(len(lines[0])):
        weightage[i]=(1/((1+b*entropy[i])**c))
    # print([reqPrototype,entropy,weightage], len(reqPrototype), len(weightage), len(entropy))
    return [reqPrototype,entropy,weightage]


