import ml
import myconstants
from myconstants import get_constant

mode = myconstants.Mode.MODE_D2V
# ml.test_similarity(stop_at=1000, max_count=3, test=True, mode=mode)

learning_rate = 0.1
ml.tf_mean_train(learning_rate=learning_rate, test=False, mode=mode, loops=5000)

