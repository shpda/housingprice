
# nnsearch.py
# nearest neighbor search

import faiss
import numpy as np

def getMatrix(feature):
    v = np.zeros((feature.shape[0], 1024))
    for i in range(feature.shape[0]):
        v[i] = feature[i]
    return v

def nnsearch(idxFeature, queryFeature, idxLabel, queryLabel, queryExpansion = 1):
    dimension = len(idxFeature[0])
    print('feature dimension = %d' % dimension)
    index = faiss.IndexFlatL2(dimension)
    #idxFeature = getVector(idxFeature)
    #queryFeature = getVector(queryFeature)
    res = faiss.StandardGpuResources()
    #res.setTempMemory(512 * 1024 * 1024)
    res.setTempMemory(512 * 2048 * 2048)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    #gpu_index.add(np.ceil(idxFeature))
    gpu_index.add(idxFeature)
    print('total num index = %d' % gpu_index.ntotal)

    for itr in range(queryExpansion):
        print('query expansion itr: %d' % itr)
        #_, neighborMatrix = gpu_index.search(np.ceil(queryFeature), 100)
        _, neighborMatrix = gpu_index.search(queryFeature, 100)
        #_, neighborMatrix = gpu_index.search(queryFeature, 2)
        #print(neighborMatrix.shape)

        queryFeature = idxFeature[neighborMatrix[:,:50]]
        queryFeature = np.mean(queryFeature, axis=1, keepdims=True)
        queryFeature = getMatrix(queryFeature)
        queryFeature = queryFeature.astype('float32').copy()

    label2result = {}
    for i in range(len(neighborMatrix)):
        neighbors = neighborMatrix[i]
        nbrLabels = ''
        for nbr in neighbors:
            nbrLabels += idxLabel[nbr]
            nbrLabels += ' '
        label2result[queryLabel[i]] = nbrLabels

    return label2result
