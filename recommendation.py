import numpy as np
from lightfm.datasets import fetch_movielens
from lightFM import LightFM
#fetching the data
data = fetch_movielens(min_rating=5.0)
#Printing the training and test data
print(repr(data['train']))
print(repr(data['test']))
#Construct the model
model = LightFM(loss='warp')
#Train the model
model.fit(data['train'], epochs=30, num_threads=2)

#Performing the Recommendation
def sample_recommendation(model, data, user_ids):
    
    #no of users and movies in training sample
    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        #getting the known positives	
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        #Result of Model
        scores = model.predict(user_id, np.arange(n_items))
        #Ranking the ratings order
        top_items = data['item_labels'][np.argsort(-scores)]
        #printing the results
        print("User %s" % user_id)
        print("     Known positives:")
        
        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")
        
        for x in top_items[:3]:
            print("        %s" % x)
        
sample_recommendation(model, data, [2, 21, 350]) 
