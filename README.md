# bigData
big data course project


1. Please download the database "reviews_Electronics_5.json.gz" in our google drive, link below

https://drive.google.com/drive/folders/0ByehYI2Lxf1nQWZTQ05LbzV2SGM

  1.) database/reviews.txt
  2.) database/train_data.json
  3.) doc2vec/Archive.zip

2. Put 1.) and 2.) in database folder and unzip 3.) into doc2vec folder where you downloaded from github repository.

4. Run tf_train.py (You can cahange the "learning_rate", "loops", "mode". Also you can change comparison sets by tweeking "CLASSES = [[1], [5]]" in myconstants.py)

5. You don't need to run preprocess.py any more.




Note: Please Keep the folder structure the same as below, otherwise you might get unexpected Errors

--Folder structure:

code
  myconstants.py
  preprocess.py
  tf_train.py
  README.md
  LICENSE
  .gitignore
--database
    train_data.json
    validate_data.json
    reviews.txt
  --test
      reviews_Electronics_5.json
      train_data.json
      validate_data.json
      reviews.txt
  --matrix
      b1_1234_5.npy
      w1_1234_5.npy
      ......
--ml
    __init__.py
    functions.py
    tf_training.py
    w2v.py
    d2v.py
--word2vec
    w2v_model
  --test
      reviews.txt
      w2v_model
      w2v_model.txt
--doc2vec
    d2v_model
    d2v_model.docvecs.doctag_syn0.npy
    d2v_model.syn0.npy
    d2v_model.syn1.npy
  --test
      d2v_model
--results
    d2v_1234_5.txt
    mean_123_45.txt
    ......
