import keras
from keras.models import *
from keras.layers import *
import json
from pprint import pprint


class SPMgr:
    def __init__(self, db_path, word_book_path):
        self.db_path = db_path
        self.word_book_path = word_book_path

        try:
            with open(word_book_path, 'r') as f:
                self.word_book = json.load(f)
        except:
            self.make_word_book(db_path, word_book_path)
            with open(word_book_path, 'r') as f:
                self.word_book = json.load(f)

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.get_db(db_path)

        self.model = self.get_model()

    def load_model(self, model_path='./semantic_parser.h5'):
        self.model = load_model(model_path)
        print(f'load model [{model_path}]')

    def test_case(self):
        questions = ['How many mugs are there in the room?',
                     'How many heads of lettuce are there in the room?',
                     'What is the relationship between mug and bread?',
                     'I think a fridge is in the room. Is that correct?',
                     'Is there a coffeemachine somewhere in the room?',
                     'Please tell me what is the thing agent have.']
        for q in questions:
            input_q = self.preprocessing(q)
            print('Q:', q)
            #print(input_q)
            p, s, o = self.model.predict(input_q[np.newaxis, :])

            p = np.argmax(p)
            s = np.argmax(s)
            o = np.argmax(o)
            #print(p, s, o)
            print('PSO:', self.answer_parsing(p, s, o))
            print('')

    def predict(self, question):
        input_q = self.preprocessing(question)
        p, s, o = self.model.predict(input_q[np.newaxis, :])

        p = np.argmax(p)
        s = np.argmax(s)
        o = np.argmax(o)

        result = self.answer_parsing(p, s, o)
        return self.triple_to_query(*result), result

    def triple_to_query(self, p, s, o):
        predicate_dict = {
            'numberOfObject':'Counting',
            'objectExistence':'Existence',
            'mainColorObject':'Attribute/Color',
            'openStateOfObject':'Attribute/Openstate',
            'objectsInsideOf':'Include',
            'objectSpatialRelation':'Relation',
            'agentHaveObject':'AgentHave'
        }
        query = predicate_dict[p]
        for d in [s, o]:
            if d != 'null':
                query += f'/{d}'
        return query

    def test_model(self):
        output = self.model.evaluate(self.test_x, [self.test_y[:,0], self.test_y[:,1], self.test_y[:,2]], batch_size=32, verbose=0)
        print(output)
        print('test done.')

    def train_model(self, model_path='./semantic_parser.h5'):
        self.model.summary()
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #

        train_output = self.model.fit(self.train_x, [self.train_y[:, 0], self.train_y[:, 1], self.train_y[:, 2]], batch_size=16, epochs=20,
                                 verbose=1, validation_split=0.2)
        print(train_output)
        self.model.save(model_path)
        print('train done.')

    def get_model(self):
        input = Input(shape=(14,), dtype='int32')
        f = Embedding(input_dim=90, output_dim=20, input_length=14)(input)
        f = Dropout(0.2)(f)
        #f = Bidirectional(LSTM(100, return_sequences=True))(f)
        f = Bidirectional(LSTM(100, return_sequences=False))(f)
        f = Dense(200, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Dropout(0.2)(f)
        feature = Dense(200, activation='relu')(f)
        predicate = Dense(7, activation='softmax')(feature)
        subject = Dense(25, activation='softmax')(feature)
        object = Dense(25, activation='softmax')(feature)

        model = Model(inputs=[input], outputs=[predicate, subject, object])

        # input = Input(shape=(14,), dtype='int32')
        # f = Embedding(input_dim=90, output_dim=20, input_length=14)(input)
        # f = Dropout(0.2)(f)
        # f = Bidirectional(LSTM(100, return_sequences=False))(f)
        # #f = SimpleRNN(20, return_sequences=False)(f)
        # f = Dense(10, activation='relu')(f)
        # predicate = Dense(7, activation='softmax')(f)
        # subject = Dense(25, activation='softmax')(f)
        # object = Dense(25, activation='softmax')(f)
        #
        # model = Model(inputs=[input], outputs=[predicate, subject, object])

        return model

    def make_word_book(self, db_path, word_book_path):
        with open(db_path, 'r') as f:
            db = json.load(f)
        nq = []
        p = []
        s = []
        o = []
        nq_word_set = set()
        p_word_set = set()
        s_word_set = set()

        for v in db.values():
            for vv in v.values():
                for vvv in vv.values():
                    nq.append(vvv['natural_question'])
                    for w in vvv['natural_question'].split():
                        nq_word_set.add(w.replace('.','').replace('\'s','').replace('?',''))

                    p.append(vvv['predicate'])
                    p_word_set.add(vvv['predicate'])

                    s.append(vvv['subject'])
                    s_word_set.add(vvv['subject'])

                    o.append(vvv['object'])
                    s_word_set.add(vvv['object'])
        print(s_word_set)
        s_word_set.remove(None)
        nq_word_dict = {word: i+1 for i, word in enumerate(nq_word_set)}
        p_word_dict = {word: i for i, word in enumerate(p_word_set)}
        s_word_dict = {word: i+1 for i, word in enumerate(s_word_set)}
        nq_word_dict['EOS'] = 0
        s_word_dict[None] = 0
        word_book = {
            'question_word':nq_word_dict,
            'predicate':p_word_dict,
            'object':s_word_dict
        }
        with open(word_book_path, 'w') as f:
            json.dump(word_book, f, indent='\t')
            print(f'save word_book [{word_book_path}]')

    def get_db(self, db_path, test_rate=0.1):
        with open(db_path, 'r') as f:
            db = json.load(f)
        nq = []
        p = []
        s = []
        o = []

        nq_word_dict = self.word_book['question_word']
        p_word_dict = self.word_book['predicate']
        s_word_dict = self.word_book['object']

        for v in db.values():
            for vv in v.values():
                for vvv in vv.values():
                    nq.append(vvv['natural_question'])
                    p.append(vvv['predicate'])
                    s.append(vvv['subject'])
                    o.append(vvv['object'])

        print(len(nq_word_dict))

        print(len(p_word_dict))
        print(len(s_word_dict))
        len_train = int(len(nq)*(1-test_rate))
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        max_len = 0
        for i in range(len(nq)):
            if i < len_train:
                input = self.preprocessing(nq[i])
                train_x.append(input)

                train_y.append([p_word_dict[p[i]], s_word_dict.get(s[i], 0), s_word_dict.get(o[i], 0)])

            else:
                input = self.preprocessing(nq[i])
                test_x.append(input)
                test_y.append([p_word_dict[p[i]], s_word_dict.get(s[i], 0), s_word_dict.get(o[i], 0)])
        print('max_len', max_len)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        return (train_x, train_y), (test_x, test_y)

    def preprocessing(self, question, question_length=14):
        input = []
        for w in question.split():
            w = w.replace('.', '').replace('\'s', '').replace('?', '')
            input.append(self.word_book['question_word'][w])
        while len(input) != question_length:
            if len(input) > question_length:
                input = input[:question_length]
            else:
                input.append(0)
        return np.array(input)

    def answer_parsing(self, p, s, o):
        with open('./word_book.json', 'r') as f:
            wb = json.load(f)
        pd_i2s = {v: k for k, v in wb['predicate'].items()}
        obj_i2s = {v: k for k, v in wb['object'].items()}

        return pd_i2s[p], obj_i2s[s], obj_i2s[o]

if __name__ == '__main__':
    test = False

    spmgr = SPMgr(db_path='/home/ailab/DH/ai2thor/datasets/190514 gsg_gt/qa_scenario.json',
                  word_book_path='./word_book.json')

    spmgr.train_model(model_path='./test_semantic_parser.h5')
    spmgr.test_model()
    #spmgr.load_model()
    #spmgr.test_case()
    #result = spmgr.predict('How many garbagecans are there in the room?')
    #print(result)


