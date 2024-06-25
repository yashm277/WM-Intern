from modifiedNMW import modifiedAlignment


def entropyBasedRuntimeMatching(reqPrototype, liveRequest, weightage_array):
    prototype  = reqPrototype
    request = liveRequest
    weightage  = weightage_array
    first = modifiedAlignment(prototype, request, weightage)
    # smin(prototype)
    second = modifiedAlignment(prototype, prototype, weightage)
    # # smax(prototype)
    # # Where ∅ is a special symbol, different to all of the character in the consensus prototype.
    # --------
    specialString = "".join(["&"] * len(prototype))
    third = modifiedAlignment(prototype,specialString, weightage)
    # --------
    relativeDistance = 1-((first-third)/(second-third))
    # print("relative distance:",relativeDistance)
    print("first", first, "second", second,"third", third)
    print("RELATIVE DISTANCE WITH", reqPrototype, "IS", relativeDistance)
    return relativeDistance

if __name__ == "__main__":
    # Paper Data
    # weightage_array = [1, 1, 1, 1, 1/1342, 1/4760, 1/4760, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1/14638, 1/14638, 1/796, 1/4760, 1/4760, 1/1342, 1/4760,1]
    # entropyBasedRuntimeMatching("{id:???,op:S,sn:???????}", "{id:37,op:A,sn:Durand}", weightage_array)
    
    # My data
    # weightage_array_1 = [1.0, 1.0, 1.0, 1.0, 0.01729507422573702, 0.0007448184988410461, 0.0007448184988410461, 0.00021007826434624494, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.831724124461783e-05, 6.831724124461783e-05, 6.831724124461783e-05, 0.00021007826434624494, 0.01729507422573702, 0.0012561786168146397, 0.0012561786168146397, 0.0007448184988410461, 0.00021007826434624494, 0.005821170289059088]
    # entropyBasedRuntimeMatching("{id:???,op:S,sn:???????", "{id:13,op:S,sn:Versteeg}" ,weightage_array_1)
    # weightage_array_2 = [1.0, 1.0, 1.0, 1.0, 0.007257755004291363, 0.007257755004291363, 0.0006035008471193843, 0.0006035008471193843, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.0006035008471193843, 0.007257755004291363, 0.0006035008471193843, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.0006035008471193843, 0.007257755004291363, 0.0006035008471193843, 0.007257755004291363, 0.0006035008471193843, 0.007257755004291363, 0.007257755004291363, 0.0006035008471193843, 0.0006035008471193843, 0.0006035008471193843, 0.0006035008471193843, 0.007257755004291363, 0.0006035008471193843, 0.007257755004291363, 0.0006035008471193843, 0.0006035008471193843, 0.007257755004291363]
    # entropyBasedRuntimeMatching("{id:????,op:A,sn:????????????????????","{id:13,op:S,sn:Versteeg}",weightage_array_2)
    # weightage_array_3 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.0006035008471193843, 0.0006035008471193843, 0.0006035008471193843, 0.007257755004291363, 0.007257755004291363, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363, 0.007257755004291363]
    # entropyBasedRuntimeMatching("{id:0,petId:0,quantity:0,shipDate:20240318T03:27:50.776Z,status:????????,complete:?????}","{id:0,category_id:0,category_name:string,name:doggie,photoUrls_0:string,tags_0_id:0,tags_0_name:string,status:available}",weightage_array_3)
    # weightage_array_4 = [1.0, 1.0, 1.0, 0.011541386287916098, 0.011541386287916098, 0.011541386287916098, 0.011541386287916098, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0008022187637936, 0.0008022187637936, 0.0008022187637936, 0.00016701668139897474, 0.0008022187637936, 0.00016701668139897474, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0008022187637936, 0.0008022187637936, 0.0008022187637936, 0.0051647925520448635, 0.0008022187637936, 0.0008022187637936, 0.0008022187637936, 0.0051647925520448635, 0.0051647925520448635]
    # entropyBasedRuntimeMatching("{id:????,category_id:0,category_name:string,name:??????,photoUrls_0:string,tags_0_id:0,tags_0_name:string,status:?????????}","{id:0,category_id:0,category_name:string,name:doggie,photoUrls_0:string,tags_0_id:0,tags_0_name:string,status:available}",weightage_array_4)
    
    # Problematic Data
    weightage_array_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0051647925520448635, 0.0051647925520448635, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    request_prototype_1 =   '{"query_petId":"??","headers_Accept":"application/json"}'
    weightage_array_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    request_prototype_2 =  '{"query_petId":"&&","headers_Accept":"application/json"}'
    live_request = {"query_petId":"&&","headers_Accept":"application/json"}
    # print("FIRST")
    # print(entropyBasedRuntimeMatching(request_prototype_1, live_request, weightage_array_1))
    # print("SECOND")
    # print(entropyBasedRuntimeMatching(request_prototype_2, live_request, weightage_array_2))