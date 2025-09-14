import numpy as np
import sys
sys.path.insert(0, "./TICC")
from TICC.TICC_solver import TICC
from multiprocessing import Pool
import collections
from sklearn.mixture import GaussianMixture


class DilatedTICC(TICC):
    """
    Adopted train function form the TICC algorithm to skip frames who have an overlap according the hop size of the
    feature extraction pipeline.
    """
    def __init__(self, window_size: int = 5, number_of_clusters: int = 2, lambda_parameter: float = 11e-2,
                 beta: int = 400, maxIters: int = 1000, dilation: int = 1, **kwargs):
        super(DilatedTICC, self).__init__(window_size=window_size, number_of_clusters=number_of_clusters,
                                          lambda_parameter=lambda_parameter, beta=beta, maxIters=maxIters, **kwargs)
        self.dilation = dilation
        self.iteration_log = dict()

    def stack_training_data(self, Data, n, num_train_points, training_indices):
        complete_D_train = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            for k in range(self.window_size):
                if i + (k * self.dilation) < len(Data):
                    idx_k = training_indices[i + (k * self.dilation)]
                    complete_D_train[i][k * n:(k + 1) * n] = Data[idx_k][0:n]
        return complete_D_train

    def fit(self, input_file):
        """
        Adapted version of the TICC solver that allows a dilation between windows to reduce redundancy due to
        overlapping windows
        """
        assert self.maxIters > 0, "Error, needs at least 1 iteration"

        # Get data into proper format
        times_series_arr, time_series_rows_size, time_series_col_size = self.load_data(input_file)

        # Train test split
        training_indices = np.arange(0, time_series_rows_size - (self.window_size - 1) * self.dilation)
        num_train_points = len(training_indices)

        # Stack the training data
        complete_D_train = self.stack_training_data(times_series_arr, time_series_col_size, num_train_points,
                                                    np.arange(0, time_series_rows_size))

        return self._fit_from_original(complete_D_train, time_series_col_size, time_series_rows_size, training_indices)

    def _fit_from_original(self, complete_D_train: np.ndarray, time_series_col_size: int, time_series_rows_size: int,
                           training_indices: np.ndarray):
        # Initialization
        gmm = GaussianMixture(n_components=self.number_of_clusters, covariance_type="full")
        gmm.fit(complete_D_train)
        clustered_points = gmm.predict(complete_D_train)

        self.iteration_log[0] = clustered_points.copy()

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # Perform clustering
        pool = Pool(processes=self.num_proc)
        for iters in range(self.maxIters):
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(list)
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, time_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance
            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': complete_D_train,
                                  'time_series_col_size': time_series_col_size}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(list)
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                        counter = (counter + 1) % len(valid_clusters)

                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = complete_D_train[
                                                                                              point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                  (self.window_size - 1) * time_series_col_size:self.window_size * time_series_col_size]

            if np.array_equal(old_clustered_points, clustered_points):
                break
            old_clustered_points = before_empty_cluster_assign
            self.iteration_log[iters + 1] = clustered_points.copy()
            # end of training
        if pool is not None:
            pool.close()
            pool.join()

        return clustered_points, train_cluster_inverse
