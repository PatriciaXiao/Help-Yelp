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
        self.valid_y = valid_y.tolist()
        self.n_valid = len(valid_x)

        self.test = self.IOtools.load_test()
        self.n_test = len(self.test)

        non_zero_scores_x, non_zero_scores_y = self.data.nonzero()
        self.data_nonzero = csr_matrix((np.ones(len(non_zero_scores_x)), (non_zero_scores_x, non_zero_scores_y)), shape=(self.n_users, self.n_business))

        user_id, user_star = self.IOtools.load_users_stars()
        self.user_avgstars = {self.u_idx_mapping[ user_id[i] ]: user_star[ i ] for i in range(len(user_id))}


    def similarity(self):
        similarity = cosine_similarity(self.data.transpose()) # item level similarity
        self.similarity = csr_matrix(similarity)
    
    def to_array(self, mat_1d):
        return np.asarray(mat_1d).reshape(-1)
    def predict_single(self, uidx, bidx):
        # print self.data[uidx, :].shape
        # print self.similarity[bidx, :].shape #[0, 1] range
        # b_similarity = self.similarity[bidx, :].toarray()
        b_similarity = self.to_array(self.similarity[bidx, :].toarray())
        u_score = self.to_array(self.data[uidx, :].toarray())
        u_nonzero = self.to_array(self.data_nonzero[uidx, :].toarray())
        sum_score = np.sum(np.dot(b_similarity, u_score))
        sum_norm = np.sum(np.dot(b_similarity, u_nonzero))
        if sum_norm: # then sum_score > 0
            pred = sum_score / sum_norm
        else:
            # sum_norm = 0
            # no user record on the similar items
            pred = self.user_avgstars[uidx]
        return pred

    def validate(self):
        valid_users = [self.u_idx_mapping[uid]for uid in self.valid_x["user_id"]]
        valid_bussiness = [self.b_idx_mapping[bid] for bid in self.valid_x["business_id"]]
        self.valid_pred = [self.predict_single(valid_users[i], valid_bussiness[i]) for i in range(self.n_valid)]
        # print self.valid_y
        # print max(self.valid_pred)
        self.valid_score = self.IOtools.evaluate(self.valid_pred, self.valid_y)
        print("RMSE score on validation set: {0}".format(self.valid_score))

    def prediction(self):
        test_users = [self.u_idx_mapping[uid]for uid in self.test["user_id"]]
        test_bussiness = [self.b_idx_mapping[bid] for bid in self.test["business_id"]]
        self.test_pred = [self.predict_single(test_users[i], test_bussiness[i]) for i in range(self.n_test)]
        self.IOtools.write_file(self.test_pred, "standardCF.csv")



print("building the model")
model = CF()
print("loading data")
model.load_data()
print("computing similarity")
model.similarity()
print("running on validation set")
model.validate()
print("getting test result")
model.prediction()



