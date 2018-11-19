############################################
# Standard Collaborative Filtering
# a simple version
# neighborhood method,
#     item-based
# planning on implementing latent factor method latter
#      Patricia Xiao at 11.18, 2018
############################################

from io_utils import *
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "../dataset/"
OUT_DIR = "./"

class CF:
    def __init__(self):
        self.IOtools = IO(data_directory=DATA_DIR)
        self.valid_pred = None
        self.predict = None
    def load_data(self):
        self.users = self.IOtools.load_users().user_id.unique() # len: 41720 # numpy ndarray
        self.business = self.IOtools.load_business().business_id.unique() # len: 12058
        self.n_users = len(self.users)
        self.n_business = len(self.business)
        self.u_idx_mapping = {self.users[idx]:idx for idx in range(self.n_users)} # id -> idx
        self.b_idx_mapping = {self.business[idx]:idx for idx in range(self.n_business)}

        uid_list, bid_list, label_list = self.IOtools.load_train_simple()
        uidx_list = [self.u_idx_mapping[uid] for uid in uid_list]
        bidx_list = [self.b_idx_mapping[bid] for bid in bid_list]
        self.data = csr_matrix((label_list, (uidx_list, bidx_list)), shape=(self.n_users, self.n_business))

        valid_x, valid_y = self.IOtools.load_valid()
        self.valid_x = valid_x
        self.valid_y = np.array(valid_y)
        self.n_valid = len(valid_x)

        self.test = self.IOtools.load_test()

    def similarity(self):
        similarity = cosine_similarity(self.data.transpose(), self.data.transpose()) # item level similarity
        # print len(similarity)
        self.similarity = csr_matrix(similarity)
        # print self.similarity
        # print self.similarity.transpose()
    def prediction(self):
        self.predict = self.data.dot(self.similarity)
        self.normalizer = np.sum(self.similarity, axis=1)
        self.normalizer = [n if n else 1.0 for n in self.normalizer]
        # self.predict = self.predict.todense() / normalizer
        # print np.max(self.predict)

    def validate(self):
        valid_users = [self.u_idx_mapping[uid]for uid in self.valid_x["user_id"]]
        valid_bussiness = [self.b_idx_mapping[bid] for bid in self.valid_x["business_id"]]
        self.valid_pred = np.array([self.predict[valid_users[i], valid_bussiness[i]] / self.normalizer[valid_bussiness[i]] for i in range(self.n_valid)])
        self.valid_score = self.IOtools.evaluate(np.array(self.valid_pred), np.array(self.valid_y))
        print self.valid_score[0][0]



model = CF()
model.load_data()
model.similarity()
model.prediction()
model.validate()



