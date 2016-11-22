from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import json

from myconstants import get_constant


class LabeledLineSentence(object):
  def __init__(self, filelist):
    self.filelist = filelist

  def __iter__(self):
    for file in self.filelist:
      for uid, line in enumerate(open(file[0])):
        if len(file) > 2 and file[2] == 'JSON':
          line = line.strip()
          data = json.loads(line)
          line = data['reviewText']
        yield LabeledSentence(words=line.split(), tags=['{}_{}'.format(file[1], uid)])


def d2v_train(test=True, combine=False):
  train_path = get_constant('D2V_SRC_PATH', test)
  accuracy_path = get_constant('ACCURACY_PATH', test)
  model_path = get_constant('D2V_MODEL_PATH', test)

  if combine:
    sentences = LabeledLineSentence([[train_path, 'TRAIN'], [accuracy_path, 'TEST', 'JSON']])
  else:
    sentences = LabeledLineSentence([[train_path, 'TRAIN']])

  model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
  model.build_vocab(sentences)
  for epoch in range(5):
    model.train(sentences)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

    model.save(model_path)


def d2v_sentence(model, sentence, datatype='NEW'):
  length = len(model.docvecs)
  sen_name = '{}_{}'.format(datatype, length)
  sentences = [LabeledSentence(words=sentence.split(), tags=[sen_name])]
  model.train(sentences)

  return model[sen_name]


# model_path = '../doc2vec/test/d2v_model'
# model_loaded = Doc2Vec.load(model_path)
# print len(model_loaded.vocab)
# print model_loaded.docvecs['1_1']

