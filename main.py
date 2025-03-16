from flask import Flask, jsonify, request
from flask_cors import CORS
from logic.recommender import Recommender
from models.anime import Anime

app = Flask(__name__)
CORS(app)

# initializing the recommendation engine
recommender = Recommender()

'''
Endpoint to retrieve data in a paginated way from recommender.data
'''
@app.route('/', methods=['GET'])
def get_data():
    # retrieve optional params
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    query = request.args.get('query', '')

    # filtering by query
    filtered_data = recommender.original_data[recommender.original_data['name'].str.contains(query)]

    # pagination
    start = (page - 1) * limit
    end = start + limit
    paginated_data = filtered_data[start:end]

    anime_objects = [
        Anime(record).to_json() for record in paginated_data.to_dict('records')
    ]

    return jsonify({
        'page': page,
        'limit': limit,
        'total': len(filtered_data),
        'data': anime_objects
    })


if __name__ == '__main__':
    app.run(debug=True)
