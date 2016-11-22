1. Please download the database "reviews_Electronics_5.json.gz" with the link below

http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz

2. Unzip it in database folder.

3. Run preprocess.py. (You can flip the 'test=True or test=False' to run test or production)

4. Run mean_train.py (You can change 'max_count' to grab different sets of comparison data)

Note: Please Keep the folder structure, otherwise you might get unexpected Errors




--Folder structure:

code
  myconstants.py
  preprocess.py
  mean_train.py
  README.txt
--database
    reviews_Electronics_5.json
    test_database.json
    train_data.json
    validate_data.json
  --test
      reviews_Electronics_5.json
      train_data.json
      validate_data.json
--ml
    __init__.py
    functions.py
    tf_mean.py
    w2v.py
--word2vec
    reviews.txt
    w2v_model
    w2v_model.syn0.npy
    w2v_model.syn1neg.npy
    w2v_model.txt
  --test
      reviews.txt
      w2v_model
      w2v_model.txt
