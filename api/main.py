from fastapi import FastAPI

from src.serving.recommender import Recommender

app = FastAPI()

recommender = Recommender()

@app.get('/recommend/{user_id}')
def recommend(user_id: int):
    print("Recommending for user {}".format(user_id))
    items = recommender.recommend(user_id)

    return {
        "user_id": user_id,
        "recommendations": items
    }