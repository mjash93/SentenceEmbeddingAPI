from EmbeddedApp import createApp
from App_Modules.EmbeddedAppModules import checkInput, computeCosineSimilarity
import pytest
import numpy as np

@pytest.fixture
def client():
    app = createApp()
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client

########################################################################
########################################################################
#                               Check the modules

def test_checkInputs(client):
    """
        Test function to check that incorrect values are caught as they pass through checkInputs
    """

    # If a list is not passed in (relevant for 2nd, 3rd endpoints)
    input = 'hello how are you doing'
    with pytest.raises(TypeError) as error:
        checkInput(input)
    assert error.type is TypeError

    # check that values are strings
    input = [None, 2, 4, '5', 4]
    with pytest.raises(TypeError) as error:
        checkInput(input)
    assert error.type is TypeError

    # check that the values are not blank spaces
    input = ['','hello','how are you']
    with pytest.raises(ValueError) as error:
        checkInput(input)
    assert error.type is ValueError

    # check that each value passed in the array is a words length sentence (greater than 1 word)
    input = ['hello how are you','you are doing well','hello_world']
    with pytest.raises(ValueError) as error:
        checkInput(input)
    assert error.type is ValueError

def test_computeCosineSimilarity(client):
    """
        Checks to ensure that the output of computeCosineSimilarity is of correct format and type
    """ 
    v1 = np.random.randn(1,5)
    v2 = np.random.randn(1,5)
    cosSim = computeCosineSimilarity(v1, v2)

    assert isinstance(cosSim, np.float64) is True 

########################################################################
########################################################################
#                               Check each endpoint for expected output format

def test_get_endpoint_success(client):
    """"
        Check overall endpoiint correctness
    """
    new_sentence = {'sentence':'The fox jumped higher this time.'}
    response = client.get('/embeddings',json=new_sentence)
    response_data = response.get_json()

    assert len(response_data) == 1
    assert 'embedding' in response_data.keys()
    assert isinstance(response_data['embedding'],list) is True
    for item in response_data['embedding']:
        assert isinstance(item, float) is True
    assert response.status_code == 200

def test_get_endpoint_key_error(client):
    """
        Test to ensure that required keys are assed. 
    """
    new_sentence = {'sentences': 'The fox jumped higher this time.'}
    response = client.get('/embedding',json=new_sentence)
    assert response.status_code == 404

def test_bulk_post_endpoint_success(client):
    """
        Test to check correctness for bulk uploads
    """
    new_bulk = {
        'sentences': [
            'This is my first string.',
            'This is my second string.'
        ]
    }
    response = client.post(
        '/embeddings/bulk',
        json=new_bulk,
        content_type = 'application/json')
    response_data = response.get_json()

    assert len(response_data) == 1
    assert 'embeddings' in response_data.keys()
    assert isinstance(response_data['embeddings'],list) is True
    for inner_list in response_data['embeddings']:
        assert isinstance(inner_list, list) is True
        for item in inner_list:
            assert isinstance(item, float) is True
    assert response.status_code == 200

def test_bulk_post_endpoint_key_error(client):
    """
        Test function to ensure correct error code is thrown for missing key (required)
    """
    new_bulk = {
        'sentence': [
            'This is my first string.',
            'This is my second string.'
        ]
    }
    response = client.post(
        '/embeddings/bulk',
        json=new_bulk,
        content_type = 'application/json')
    assert response.status_code == 400

def test_similarity_post_endpoint_success(client):
    """
        Test to check for endpoint correctness
    """
    new_bulk = {
        'sentence_1':' The dog went over the moon',
        'sentence_2': 'How are you doing this fine morning'
    }
    response = client.post(
        '/embeddings/similarity',
        json=new_bulk,
        content_type='application/json')
    response_data = response.get_json()

    assert len(response_data) == 1
    assert 'similarity' in response_data.keys()
    assert isinstance(response_data['similarity'], float) is True
    assert response.status_code == 200

def test_similarity_post_endpoint_key_error(client):
    """
        Test function to ensure correct error code is thrown for missing keys
    """
    new_bulk = {
        'sentence1':'The dog went over the moon',
        'sentence_2': 'How are you doing this fine morning'
    }
    response = client.post(
        '/embeddings/similarity',
        json=new_bulk,
        content_type='application/json')
    assert response.status_code == 400