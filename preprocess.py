import logging
import ml

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# # 1. Extract reviews and ratings from original json file
# # Within this step, you will get 'train_data.json', 'validate_data.json', 'reviews.txt'
# ml.scrape(test=True, train=1.0)

# # 2. Train review text to vectors
# ml.w2v_train(test=True)

# 3. Train review sentenses to vectors
# ml.d2v_train(test=False, combine=True)
