import numpy as np

np.random.seed(1234)
class RatingsData(object):
    def __init__(self, raw_data_path):
        with open(raw_data_path, 'r') as f:
            content = f.read()

        self._ratings_data = {}

        for line in content.split('\n'):
            if not line:
                continue

            values = line.split('::')
            user_ratings = self._ratings_data.get(values[0], {})
            user_ratings[values[1]] = float(values[2])
            self._ratings_data[values[0]] = user_ratings

    def as_matrix(self, movies):

        sorted_users = sorted(self._ratings_data.keys())

        matrix = []

        for user_id in sorted_users:
            row = []
            for movie in movies:
                row.append(self._ratings_data[user_id].get(movie, np.NaN))
            matrix.append(row)

        return np.array(matrix)


class MoviesData(object):
    def __init__(self, raw_data_path):
        with open(raw_data_path, 'r') as f:
            content = f.read()

        self._movies_data = {}

        for line in content.split('\n'):
            if not line:
                continue

            values = line.split('::')
            self._movies_data[values[0]] = {
                'name': values[1],
                'genres': values[2].split('|')
            }

    def get_available_movies(self):
        return sorted(self._movies_data.keys())


def split_train_test(ratings_matrix, ratio=0.8):
    train = []
    test = []

    for i, user_ratings in enumerate(ratings_matrix):
        known_ratings = np.argwhere(~np.isnan(user_ratings)).flatten()
        train_ratings = np.random.choice(known_ratings, int(len(known_ratings) * ratio), replace=False)
        test_ratings = np.setdiff1d(known_ratings, train_ratings)

        row = user_ratings.copy()
        mask = np.ones(row.shape, dtype=bool)
        mask[train_ratings] = False
        row[mask] = np.NaN
        train.append(row)

        row = user_ratings.copy()
        mask = np.ones(row.shape, dtype=bool)
        mask[test_ratings] = False
        row[mask] = np.NaN
        test.append(row)

    return np.vstack(train), np.vstack(test)


class Hyperparameters(object):

    def __init__(self, dim, reg_users=1e-1, reg_items=1e-1, reg_bias_users=1e-1, reg_bias_items=1e-1):
        self.k = dim
        self.lambda_u = reg_users
        self.lambda_v = reg_items
        self.lambda_bu = reg_bias_users
        self.lambda_bv = reg_bias_items


class MFModel(object):

    def __init__(self, hyperparams, num_of_users, num_of_movies):
        self.hyper = hyperparams
        self.U = np.random.randn(hyperparams.k, num_of_users)
        self.V = np.random.randn(hyperparams.k, num_of_movies)
        self.Ub = np.random.randn(num_of_users, 1)
        self.Vb = np.random.randn(num_of_movies, 1)


def derive_by_um(R, U, V, Ub, Vb, lambda_u, m):
    predict = np.vectorize(lambda n: np.dot(U.T[m], V[:, n]) + Vb[n] + Ub[m])
    ns = np.argwhere(~np.isnan(R[m])).reshape(-1)
    error = (R[m, ns] - predict(ns)).reshape(-1, 1) * -V[:, ns].T

    return error.sum(axis=0) + lambda_u * U[:, m]


def derive_by_vn(R, U, V, Ub, Vb, lambda_v, n):
    predict = np.vectorize(lambda m: np.dot(U.T[m], V[:, n]) + Vb[n] + Ub[m])
    ms = np.argwhere(~np.isnan(R[:, n])).reshape(-1)
    error = (R[ms, n] - predict(ms)).reshape(-1, 1) * -U[:, ms].T

    return error.sum(axis=0).reshape(-1) + lambda_v * V[:, n]


def derive_by_bm(R, U, V, Ub, Vb, lambda_bu, m):
    predict = np.vectorize(lambda n: np.dot(U.T[m], V[:, n]) + Vb[n] + Ub[m])
    ns = np.argwhere(~np.isnan(R[m]))
    error = -(R[m, ns] - predict(ns))

    return error.sum() + lambda_bu * Ub[m]


def derive_by_bn(R, U, V, Ub, Vb, lambda_bv, n):
    predict = np.vectorize(lambda m: np.dot(U.T[m], V[:, n]) + Vb[n] + Ub[m])
    ms = np.argwhere(~np.isnan(R[:, n]))
    error = -(R[ms, n] - predict(ms))

    return error.sum() + lambda_bv * Vb[n]


class SGDParameters(object):
    def __init__(self, step_size=1e-3, epocs=50):
        self.alpha = step_size
        self.epocs = epocs


def RMSE(U, V, Ub, Vb, dataset):
    pred = np.dot(U.T, V) + Ub.reshape(-1, 1) + Vb.reshape(-1, 1).T
    mask = ~np.isnan(dataset)
    return np.sqrt(np.power(pred[mask] - dataset[mask], 2).sum() / mask.sum())


def cost(dataset, U, V, Ub, Vb, U_reg, V_reg, Ub_reg, Vb_reg):

    predictions = np.dot(U.T, V) + Ub.reshape(-1, 1) + Vb.reshape(-1, 1).T
    mask = ~np.isnan(dataset)
    errors = np.power(predictions[mask] - dataset[mask], 2).sum()
    v_reg = V_reg * np.power(np.linalg.norm(V, axis=0), 2).sum()
    u_reg = U_reg * np.power(np.linalg.norm(U, axis=0), 2).sum()
    ub_reg = Ub_reg * np.power(Ub, 2).sum()
    vb_reg = Vb_reg * np.power(Vb, 2).sum()

    return 0.5 * (errors + v_reg + u_reg + ub_reg + vb_reg)


def LearnModelFromDataUsingSGD(dataset, model_params, algorithm_params):
    U, V, Ub, Vb = model_params.U, model_params.V, model_params.Ub, model_params.Vb

    for epoc in range(algorithm_params.epocs):
        print 'Epoch: ', epoc
        print 'Cost: ', cost(dataset, U, V, Ub, Vb, model_params.hyper.lambda_u, model_params.hyper.lambda_v,
                             model_params.hyper.lambda_bu, model_params.hyper.lambda_bv)
        print 'RMSE: ', RMSE(U, V, Ub, Vb, dataset)

        for m, n in np.argwhere(~np.isnan(dataset)):

            U[:, m] -= algorithm_params.alpha * derive_by_um(dataset, U, V, Ub, Vb, model_params.hyper.lambda_u, m)
            V[:, n] -= algorithm_params.alpha * derive_by_vn(dataset, U, V, Ub, Vb, model_params.hyper.lambda_v, n)
            Ub[m] -= algorithm_params.alpha * derive_by_bm(dataset, U, V, Ub, Vb, model_params.hyper.lambda_bu, m)
            Vb[n] -= algorithm_params.alpha * derive_by_bn(dataset, U, V, Ub, Vb, model_params.hyper.lambda_bv, n)

    return U, V, Ub, Vb


reduced_dataset = np.load('dataset.npy')
model_params = MFModel(Hyperparameters(100), *reduced_dataset.shape)
algorithm_params = SGDParameters()
params = LearnModelFromDataUsingSGD(reduced_dataset, model_params, algorithm_params)