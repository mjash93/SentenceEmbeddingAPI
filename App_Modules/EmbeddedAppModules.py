import re
import numpy as np
from scipy import spatial

# House functions to 
#   A) Check for incinsistences in data input
#   B) run model and reformat results for JSON output

def computeCosineSimilarity(vec1, vec2):
    """
        Compute the cosine similarity between two numeric N-dimensional vectors. Uses the SciPy library. 
    """
    return spatial.distance.cosine(vec1, vec2)

def runModel(model, inputs):
    """
        RunModel runs the embedded TF model and returns the resulting embeddings as a JSON dictionary as determined by challenge parameters. 

        Input:
            model -> embedded tensorflow model
            inputs -> list
    """
    embeddings = model(inputs).numpy()
    return embeddings

def checkInput(inputs):
    """
        Checks the inputs for any issues. If an issue is found, raise a specific error. Assumption is that the request will parse everything to a string. 
    """
    if isinstance(inputs, list) == False:
        raise TypeError(f'The input parameter of {type(inputs)} is incorrect. A list needs to be passed.')
    N = len(inputs)

    for i in range(N):
        if isinstance(inputs[i], str) == False:
            raise TypeError(f'The {i}th input type of {type(input[i])} is incorrect. Please pass in a string.')
        # check to make sure empty string not passed in
        if len(inputs[i]) == 0:
            raise ValueError('Sentence input must be of length greater than 0.')
        # checks to ensure that the sentence is more than a signle word, as defined by spaces between words
        sentenceLength = len(re.split('\s',inputs[i]))
        if sentenceLength < 2:
            raise ValueError('Sentence input must be greater than {}.'.format(sentenceLength))

def processBulkUpload(self, sentences):
    checkInput(sentences)
    results = runModel(self.embed, sentences)
    return results.tolist()

def processSimilarityUpload(self, sentences):
    checkInput(sentences)
    results = runModel(self.embed, sentences)
    vec1, vec2 = np.split(results, 2)
    similarityScore = computeCosineSimilarity(vec1, vec2)
    return similarityScore


def processPostRequest(self, inputs,endpoint='bulk'):
    """
        Top level method to handle processing and checks for the POST method
    """
    output = {}
    try:
        if endpoint == 'bulk':
            sentences = inputs['sentences']
            output['embeddings'] = processBulkUpload(self, sentences)
        elif endpoint == 'similarity':
            sentence1 = inputs['sentence_1']
            sentence2 = inputs['sentence_2']
            output['similarity'] = processSimilarityUpload(self,[sentence1, sentence2])
        else:
            raise Exception(f'Endpoint {endpoint} is currently not supported.')
    except Exception as e:
        output['Message'] = e.args
        return output, 400
    else:
        return output, 200

def processGetRequest(self, input):
    """
        Top level method to handle processing and checks for the GET endpoint -> /embeddings
    """
    output = {}
    input_list = []
    sentence = input['sentence']
    input_list.append(sentence)
    try:
        # run basic check on the sentence value
        checkInput(input_list)
        # pass the input to the model to run
        embedding = runModel(self.embed, input_list)
        output['embedding'] = embedding.tolist()[0]
    except Exception as e:
        output['Message'] = e.args
        return output, 400
    else:
        return output, 200
