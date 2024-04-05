from math import inf
import pandas as pd
import json, pickle
from basicNMW import getAlignedScore
from clustal import clustalW_MSA
from hierarchialClustering import hierarchialClustering
from requestPrototype import FormulatingtheRequestPrototype
from flatten_json import flatten, unflatten
from runtimeMatch import entropyBasedRuntimeMatching

def removeRequestNesting_for_data_loading(request):
    """
    This function takes in a request and flattens it into a dictionary. It also creates a unique_id for the request
    Since the method and path are not needed in the request, they are removed from the dictionary
    We will create tables based on the unique_id, since clustering later will only be done with those requests with the same unique ids
    
    Parameters:
    request: JSON request (dictionary type)
    
    Returns:
    request_removed: flattened JSON request (dictionary type)
    unique_id: unique_id for the request
    """
    request_removed = flatten(request["request"])
    unique_id = request_removed["method"] + request_removed["path"]
    del request_removed["method"]
    del request_removed["path"]
    return request_removed, unique_id
    
def removeResponseNesting_for_data_loading(response):
    """
    This function takes in a response and flattens it into a dictionary.
    
    Parameters:
    response: JSON response
    
    Returns:
    response_removed: flattened JSON response
    """
    request_removed = flatten(response["response"])
    return request_removed

def reconstructResponse(response):
    """
    This function takes in a response and reconstructs it into a dictionary.
    
    Parameters: response: JSON response
    
    Returns: response_reconstructed: reconstructed JSON response
    """
    return d2squotes(json.dumps({"response":unflatten(json.loads(response))}))

def reconstructRequest(request, unique_id):
    """
    This function takes in a request and reconstructs it into a dictionary.
    
    Parameters:
    request: JSON request
    unique_id: unique_id for the request
    
    Returns:
    request_reconstructed: reconstructed JSON request
    
    """
    method, path = unique_id.split('/', 1)
    path = '/' + path
    return d2squotes(json.dumps({"request":{"method":method, "path":path, **unflatten(json.loads(request))}}))

def d2squotes(string):
    """
    Making the string comparable to JSON by making it print double quotes instead of single quotes
    
    Parameters:
    string: string that needs to be converted
    
    Returns:
    string: string with double quotes
    """
    for i in range(len(string)):
        if string[i] == "'":
            string = string[:i] + '"' + string[i+1:]
    return string

def checkJSONFormatting(data):
    """
    This function is for error handling. It will check if the JSON is formatted correctly.
    If it is not, it will print out the row number and the error message.
    This was introduced to identify the error earlier, since otherwise it becomes very difficult to identify errors
    
    Parameters:
    data: initial pandas dataframe with the JSON data
    
    Returns:
    None
    """
    count=0
    for i, j in data.iterrows():
        try:
            json.loads(j.iloc[0])
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON request for row {i+1}: {e}")
            count+=1
            continue
        try:
            json.loads(j.iloc[1])
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response for row {i+1}: {e}")
            count+=1
            continue
    if count==0:
        print("\nAll JSON is formatted correctly\n")
    else:
        print(f"\nERROR: Failed to parse {count} JSON rows, please look at fixing these rows")
        return -1

def readData(path):
    """
    This function reads the data from the csv file and checks if the JSON is formatted correctly.
    If it is not, it will print out the row number and the error message.
    
    Parameters:
    path: path to the csv file
    
    Returns:
    data: pandas dataframe with the JSON data

    """
    data = pd.read_csv(path)
    checkJSONFormatting(data)
    return data

def tableCreation(data):
    """
    This function creates an dictionary based on the unique_id as a key. 
    The "value" of the dictionary is a pandas dataframe with the flattened JSON request and responses
    The tables are created by flattening the JSON request and removing the method and path.
    The unique_id is created by concatenating the method and path.
    
    COMMENTED FOR DEBUGGING PURPOSES:
    Also, it gives you a warning if you have a small table size that for a particular unique_id the prediction will be bad since your dataset is not large enough.
    You have the option to continue past the error or stop the function and add more points to your dataset.
    
    Parameters:
    data: pandas dataframe with the JSON data
    
    Returns:
    tables_dict: dictionary with the unique_id as the key and the flattened JSON request and responses corresponding to that unique_id
    """
    tables_dict = {}
    for i, j in data.iterrows():
        request_removed, unique_id = removeRequestNesting_for_data_loading(json.loads(j.iloc[0]))
        response_removed = removeResponseNesting_for_data_loading(json.loads(j.iloc[1]))
        request_removed, response_removed = json.dumps(request_removed), json.dumps(response_removed)
        if unique_id in tables_dict:
            new_row = pd.DataFrame([[request_removed, response_removed]], columns=['request', 'response'])
            tables_dict[unique_id] = pd.concat([tables_dict[unique_id], new_row], ignore_index=True)
        else:
            tables_dict[unique_id] = pd.DataFrame([[request_removed, response_removed]], columns=['request', 'response'])
    # for i in tables_dict:
    #     if tables_dict[i].shape[0] < 5:
    #         print(f"WARNING: Table size for the unique_id {i} is small. Prediction will be bad. Do you want to continue? (yes/no)")
    #         answer = input()
    #         if answer == "no":
    #             print(f"You terminated the process, come back with updated data for {i}")
    #             return -1
    return tables_dict
        
def dataCollection():
    """
    This encapsulates everything we want from the "data" folder. This function can be imported into wherever needed.
    It reads the data from the csv file, checks if the JSON is formatted correctly, and creates tables based on the unique_id.
    
    TODO:
    THE PATH FOR THE DATA COLLECTION CAN BE PASSED DIRECTLY HERE
    
    Parameters:
    None
    
    Returns:
    dictionaryofTables: dictionary with the unique_id as the key and the flattened JSON request and responses corresponding to that unique_id
    """
    path = "data/noErrors.csv"
    data = readData(path)
    dictionaryofTables = tableCreation(data)
    return dictionaryofTables

def trainingOneUniqueId(df):
    """
    This function is used to train the model using the data provided in the dataframe which corresponds to a unique ID.
    The function uses the data to form clusters of requests and then formulates the request prototype for each cluster.
    
    COMMENTING FOR DEBUGGING:
    If there is only one req, resp pair in the table then it returns a warning
    
    Parameters:
    df (DataFrame): The dataframe containing the data for a unique ID
    
    Returns:
    return [allRequestPrototypesInfoForThisUniqueID, df]
    returns the dataframe and the request prototype, entropy and weightage for each cluster in this particular unique ID.
    """
    if df.shape[0]==1:
        # print("WARNING: You only have one request in this method+path combo, it is strongly suggested you add more (ignore/exit)")
        # answer = input()
        # if answer == "exit":
        #     return -1
        onlyReqInTable = df.iloc[0,0]
        print(f"\nRequest Prototype for this cluster: {onlyReqInTable}\nEntropy: {[0]*len(onlyReqInTable)}\nWeightage: {[1]*len(onlyReqInTable)}")
        return [onlyReqInTable,[0]*len(onlyReqInTable),[1]*len(onlyReqInTable),df]
    distance_matrix = [[0]*len(df) for _ in range(len(df))]
    for index1, rowdata1 in df.iterrows():
        for index2, rowdata2 in df.iterrows():
            distance_matrix[index1][index2] = getAlignedScore(rowdata1.iloc[1], rowdata2.iloc[1])
    clusters = hierarchialClustering(distance_matrix, 400)
    print(f"Clusters of datapoints: {clusters}")
    allRequestPrototypesInfoForThisUniqueID = []
    for cluster_number, oneCluster in enumerate(clusters, start=1):
        requestsForOneCluster = []
        for index in oneCluster:
            requestsForOneCluster.append(df.iloc[index, 0])
        print(f"\nRequests in Cluster {cluster_number}: {requestsForOneCluster}")
        requestsForOneCluster = clustalW_MSA(requestsForOneCluster)
        requestPrototypeInfo = FormulatingtheRequestPrototype(requestsForOneCluster, 0.8, 1, 10)
        allRequestPrototypesInfoForThisUniqueID.append((requestPrototypeInfo[0],requestPrototypeInfo[1],requestPrototypeInfo[2], oneCluster))
        print(allRequestPrototypesInfoForThisUniqueID)
    for clusterNumber, oneRequestPrototypeInfo in enumerate(allRequestPrototypesInfoForThisUniqueID):
        print(f"\nDetails for cluster number {clusterNumber+1}")
        print(f"\nRequest Prototype for this cluster:\n {oneRequestPrototypeInfo[0]}\nEntropy: {oneRequestPrototypeInfo[1]}\nWeightage: {oneRequestPrototypeInfo[2]}\nData points index from df in the cluster: {oneRequestPrototypeInfo[3]}")
    return [allRequestPrototypesInfoForThisUniqueID, df]

def trainingEntireData():
    """
    This will create a dicitonary of all the unique_ids and their corresponding request prototypes, entropy and weightage.
    This is then stored in pickle format for future use so that we don't have to train the model again.
    
    Parameters:
    None
    
    Returns:
    allRequestPrototypesInfo (Dict): A dictionary containing the unique_id as the key and the request prototype, entropy and weightage for each cluster in that unique_id.
    """
    allRequestPrototypesInfo = {}
    dataDictionary = dataCollection()
    for unique_id, df in dataDictionary.items():
        print(f"\nTraining for unique_id: {unique_id}")
        allRequestPrototypesInfo = {**allRequestPrototypesInfo, unique_id: trainingOneUniqueId(df)}
    return allRequestPrototypesInfo

def save_to_pickle(data, filename):
    """
    Saves the given data to a pickle file.

    Parameters:
    - data: The data to be pickled.
    - filename: The name of the file where the pickled data will be stored.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(filename):
    """
    Loads and returns the data from a pickle file.

    Parameters:
    - filename: The name of the file to load the data from.
    
    Returns:
    - The data that was pickled in the given file. 

    PLEASE NOTE THIS RETURNS A DICTIONARY OF (UID, DF), NOT AN ARRAY OR ANYTHING ELSE
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

def helper_print_prototypeInfo_fromDictionary(oneRequestPrototypeInfo):
    print(f"\nRequest Prototype for this cluster: {oneRequestPrototypeInfo[0][0]}\n\nEntropy: {oneRequestPrototypeInfo[0][1]}\n\nWeightage: {oneRequestPrototypeInfo[0][2]}\n")

def list_to_dict(input_list):
    output_dict = {}
    for item in input_list:
        # Splitting each item by the colon
        key, value = item.split(":", 1)
        # Stripping leading and trailing whitespace and quotes from the key and value
        key = key.strip().strip('"')
        value = value.strip().strip('"')
        # Adding the key-value pair to the dictionary
        output_dict[key] = value
    return output_dict

def symmetricFieldIdentification(centroidReq, centroidResp, request):
    """
    Generates the response using some logic. 
    Read code for more clarity
    
    Parameters:
    centroidReq: JSON request (string)
    centroidResp: JSON response (string)
    request: JSON request (string)
    
    Returns:
    generatedResponse: JSON response (string)
    """
    print(f"\nCentroid Request: {centroidReq}\n\nCentroid Response: {centroidResp}\n\nRequest: {request}\n\n Response: {centroidResp}\n\n")
    centroidReq_dict = json.loads(centroidReq)
    centroidResp_dict = json.loads(centroidResp)
    request_dict = json.loads(request)
    
    generatedResponse_dict = centroidResp_dict.copy()

    for key1, value1 in centroidReq_dict.items():
        for key2, value2 in centroidResp_dict.items():
            print(f"Comparing {value1} with {value2}")
            if str(value1) == str(value2):
                print(f"FOUND A MATCH")
                if key2 in generatedResponse_dict:
                    print(f"Replacing {key2} with {key1} in the response ie. {value1} with {request_dict[key1]}")
                    generatedResponse_dict[key2] = request_dict[key1]
    print(f"\nCentroid Request: {centroidReq}\n\nCentroid Response: {centroidResp}\n\nRequest: {request}\n\n Response: {centroidResp}\n\n")
    # Convert the generated response dictionary back to a JSON string
    generatedResponse = json.dumps(generatedResponse_dict)
    
    return generatedResponse

def centroidIdentification(cluster, request):
    """
    This function identifies the centroid of the cluster and then uses the centroid to identify the response.
    If the centroid is the same as the request, it returns the response corresponding to the centroid.
    If the centroid is not the same as the request, it uses the symmetric field identification to identify the response.
        
    Parameters:
    data: pandas dataframe with the JSON data
    cluster: cluster of requests
    request: JSON request
    
    Returns:
    response: JSON response
    """
    request_removed_original, unique_id = removeRequestNesting_for_data_loading(request)
    request_removed_original = json.dumps(request_removed_original)
    allPrototypesAndReqRespData = load_from_pickle('allPrototypes.pkl')
    if unique_id in allPrototypesAndReqRespData:
        RequestPrototypeInfo = allPrototypesAndReqRespData[unique_id][0]
        # print(len(RequestPrototypeInfo))
        # print(RequestPrototypeInfo)
        oneRequestReqRespData = allPrototypesAndReqRespData[unique_id][1]
    score = [0]*(len(cluster))
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            if i!=j:
                # Centroid Identification on the basis of requests
                score[i] += (getAlignedScore(oneRequestReqRespData.iloc[cluster[i],0],oneRequestReqRespData.iloc[cluster[j],0]))
                score[i] += (getAlignedScore(oneRequestReqRespData.iloc[cluster[i],0], request_removed_original))*2
    # print(score)
    # print(oneRequestReqRespData.iloc[cluster[score.index(max(score))],0])
    centroid = oneRequestReqRespData.iloc[cluster[score.index(max(score))],0]
    # print("+++++++CENTROID IS:+++++++++++")
    # print(f"Request is :\n{json.dumps(request)}\n")
    # print(f"Centroid is:\n{centroid}\n")
    # print("+++++SUBSEQUENCE MATCHING+++++")
    if centroid==request_removed_original:
        print("\nYAY, your request already exists in the database, just returning it's response\n")
        return oneRequestReqRespData.iloc[cluster[score.index(max(score))],1]
    else:
        print("\nOOPS, the centroid doesn't exist in the training data, let's try to generate the response using symmetric field identification")
        return symmetricFieldIdentification(oneRequestReqRespData.iloc[cluster[score.index(max(score))],0],oneRequestReqRespData.iloc[cluster[score.index(max(score))],1], request_removed_original)

def get_cluster(request, path_to_dataset):
    """
    This function takes in a request and returns the response from the dictionary of allPrototypes.
    It generates a response if one does not already exist - if an exact match is not found it returns the exact match
    
    TODO: HAVE TO PROPERLY REPLACE THE WHITE SPACES WITH EMPTY STRINGS. RIGHT NOW IT WILL CAUSE ISSUES. ONLY WANT TO REPLACE AFTER A COLON
    
    Parameters:
    request: JSON request
    
    Returns:
    response: JSON response
    """
    request_removed, unique_id = removeRequestNesting_for_data_loading(request)
    request_removed = json.dumps(request_removed)
    # BELOW LINE NEEDS TO BE DEALT WITH (DOCSTRING)
    request_removed = request_removed.replace(" ", "")
    allPrototypesAndReqRespData = load_from_pickle('allPrototypes.pkl')
    ans=inf
    closestMatchingCluster = -1
    if unique_id in allPrototypesAndReqRespData:
        # RequestPrototypeInfo = allPrototypesAndReqRespData[unique_id][0]
        # print(len(RequestPrototypeInfo))
        # print(RequestPrototypeInfo)
        oneRequestReqRespData = allPrototypesAndReqRespData[unique_id][1]
        # print(oneRequestReqRespData)
        for oneRequestPrototypeInfo in allPrototypesAndReqRespData[unique_id][0]:
            print(len(oneRequestPrototypeInfo[0]), len(oneRequestPrototypeInfo[1]), len(oneRequestPrototypeInfo[2]), len(request_removed))
            print(f"\nREQUEST PROTOTYPE:\n{oneRequestPrototypeInfo[0]}")
            print(f"\nJSON REQ:\n{request_removed}")
            # print(f"\nWEIGHTAGE: \n {oneRequestPrototypeInfo[2]}")
            if len(oneRequestPrototypeInfo[0])<len(request_removed):
                continue
            currans = abs(entropyBasedRuntimeMatching(oneRequestPrototypeInfo[0], request_removed, oneRequestPrototypeInfo[2]))
            if currans < ans:  
                ans = currans  
                closestMatchingCluster = oneRequestPrototypeInfo
        if closestMatchingCluster==-1:   
            print("\nERROR: Your data is too long, there was no training data that was of this length. \nThis algorithm doesn't support this. Please provide more data\n\n")
        else:
            # print("The closest matching cluster is:")
            # print(closestMatchingCluster[3])
            return closestMatchingCluster
    else:
        print("\nThere is no training data corresponding to this method+path combo, please update the training data\n\n")
        return -1


req = {
  "request": {
  "method": "PUT",
  "path": "/pet",
  "query": {},
  "headers": {
  "Accept": "application/json"
  },
  "body": {
  "id": 831712398123789123878,
  "category": {
  "id": 0,
  "name": "string"
  },
  "name": "test1",
  "photoUrls": [
  "string"
  ],
  "tags": [
  {
  "id": 0,
  "name": "ya"
  }
  ],
  "status": "pending"
  }
  }
  }

print(f"\nTHE RESPONSE IS:\n{(reconstructResponse(centroidIdentification(get_cluster(req, "data/noErrors.csv")[3], req)))}\n\n")
