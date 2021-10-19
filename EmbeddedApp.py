from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from App_Modules.EmbeddedAppModules import *
import tensorflow_hub as hub

# Source of TF hub model

def createApp():
    """
        Create the App, assign API endpoints, and load the TF model
    """
    TF_FILE = 'https://tfhub.dev/google/universal-sentence-encoder/4'

    app = Flask(__name__)
    api = Api(app)

    loaded_model = hub.load(TF_FILE)

    api.add_resource(SingleSentenceEmbedding, '/embeddings',resource_class_args=(loaded_model,))  # add endpoints
    api.add_resource(MultipleSentenceEmbedding, '/embeddings/bulk',resource_class_args=(loaded_model,))  # add endpoints
    api.add_resource(SentenceSimilarity, '/embeddings/similarity',resource_class_args=(loaded_model,))  # add endpoints

    return app

class SingleSentenceEmbedding(Resource):
    """
        GET endpoint to handle single sentence embedding
    """
    def __init__(self,embed):
        self.embed = embed
    
    def get(self):
        """
            Parse the Request information for single sentence embedding
        """
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('sentence', required=True)
        args = parser.parse_args()
        # Add check for format
        # If passed, feed into function to run model and repackage results
        return processGetRequest(self, args)

class MultipleSentenceEmbedding(Resource):
    """
        POST endpoint to handle bulk sentence uploads for embedding
    """
    def __init__(self,embed):
        self.embed = embed
    
    def post(self):
        """
            Parse the Request information for the bulk upload situation
        """
        parser = reqparse.RequestParser()
        parser.add_argument('sentences',action='append',required=True)
        args = parser.parse_args()
        return processPostRequest(self, args, endpoint='bulk')

class SentenceSimilarity(Resource):
    """
        POST endpoint to handle similarity score between two embeddings. 
    """
    def __init__(self, embed):
        self.embed = embed
    
    def post(self):
        """
            Parse the Request environment for similarity score upload
        """
        parser = reqparse.RequestParser()
        parser.add_argument('sentence_1',required=True)
        parser.add_argument('sentence_2',required=True)
        args = parser.parse_args()

        return processPostRequest(self, args, endpoint='similarity')

if __name__ == '__main__':
    app = createApp()
    app.run(debug=True)