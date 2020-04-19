import numpy as np
# import sklearn.metrics.pairwise as sklpairwise
from operator import itemgetter
import xarray_util as xr_util


def compute_matrix_similarity(template_matrices, query_matrices, method='euclid_dist_naive',
                              query_matrix_indices=None, sort_descending=True):
    """

    :param template_matrices: (list of 2D numpy ndarrays)
    :param query_matrices: (list of 2D numpy ndarrays)
    :param method:
    :return:
    """

    if method == 'euclid_dist_naive':

        if type(template_matrices) is not list:
            template_matrices = [template_matrices]
        if type(query_matrices) is not list:
            query_matrices = [query_matrices]

        similarity_score_matrix = np.zeros((len(template_matrices), len(query_matrices)))

        for n_template, template_matrix in enumerate(template_matrices):

            for n_query, query_matrix in enumerate(query_matrices):
                similarity_score = np.linalg.norm(template_matrix.flatten() - query_matrix.flatten())
                similarity_score_matrix[n_template, n_query] = similarity_score

        if n_template > 1:
            similarity_scores = np.mean(similarity_score_matrix, 2)
        else:
            similarity_scores = similarity_score_matrix

    elif method == 'euclid_dist_fast':

        if type(template_matrices) is not list:
            template_matrices = [template_matrices]
        if type(query_matrices) is not list:
            query_matrices = [query_matrices]

        query_matrix_stacked = np.stack(query_matrices, axis=-1)
        num_query_matrices = np.shape(query_matrix_stacked)[2]
        query_matrix_flattened = np.reshape(query_matrix_stacked, (-1, num_query_matrices))

        similarity_score_matrix = np.zeros((len(template_matrices), len(query_matrices)))

        for n_template, template_matrix in enumerate(template_matrices):

            # simply take the difference between one template and each flattened matrix , then take the entire norm
            similarity_score = np.linalg.norm(template_matrix.flatten() - query_matrix_flattened)
            similarity_score_matrix[n_template, n_query] = similarity_score

        if n_template > 1:
            similarity_scores = np.mean(similarity_score_matrix, 2)
        else:
            similarity_scores = similarity_score_matrix

    else:
        print('Method not supported')
        similarity_scores = None

    if sort_descending:
        sort_idx = np.argsort(similarity_scores).flatten()[::-1]  # sort in descending order
    else:
        sort_idx = np.argsort(similarity_scores).flatten()

    if query_matrix_indices is not None:
        sorted_indices = query_matrix_indices[sort_idx]
    else:
        sorted_indices = None

    return similarity_scores.flatten(), sort_idx, sorted_indices


def compute_matrix_similarity_w_feedback(matrix_list, reviewed_idx, review_score,
                                         similarity_calculation_method='euclid_dist_naive',
                                         agg_method='rank', query_matrix_indices=None):

    if type(reviewed_idx) is not list:
        reviewed_idx = [reviewed_idx]
    if type(review_score) is not list:
        review_score = [review_score]

    if type(review_score) is list:
        review_score = np.array(review_score)

    not_review_matrix_list = matrix_list.copy()

    for del_idx in sorted(reviewed_idx, reverse=True):
        # delete the higher indices first (so we can delete one by one without worrying about change of index location)
        del not_review_matrix_list[del_idx]

    num_not_review_matrix = len(not_review_matrix_list)
    num_informative_ranking = len(review_score[review_score != 0])
    num_reviewed = len(review_score)

    if num_informative_ranking == 0:

        average_rank = np.ones((num_not_review_matrix, 1))
        sort_idx = np.ones((num_not_review_matrix, 1))
        sorted_indices = np.ones((num_not_review_matrix, 1))

    else:

        #ranking_matrix = np.zeros((num_informative_ranking, num_not_review_matrix))
        ranking_matrix = np.zeros((num_reviewed, num_not_review_matrix))

        """
        for n_review, (rv_score, review_matrix) in enumerate(zip(review_score, itemgetter(*reviewed_idx)(matrix_list))):

            if rv_score > 0:
                sort_descending = True
            elif rv_score < 0:
                sort_descending = False
            elif rv_score == 0:
                continue

            print('Length of not review matrix list: ' + str(len(not_review_matrix_list)))

            _, sort_idx, _, = compute_matrix_similarity(template_matrices=[review_matrix],
                                                        query_matrices=not_review_matrix_list,
                                                        method=similarity_calculation_method,
                                                        sort_descending=sort_descending)

            ranking_matrix[n_review, :] = sort_idx
        """
        for n_review, (rv_score, rv_idx) in enumerate(zip(review_score, reviewed_idx)):

            if rv_score > 0:
                sort_descending = True
            elif rv_score < 0:
                sort_descending = False
            elif rv_score == 0:
                continue

            review_matrix = matrix_list[rv_idx]

            # print('Length of not review matrix list: ' + str(len(not_review_matrix_list)))
            # print(np.shape(review_matrix))

            _, sort_idx, _, = compute_matrix_similarity(template_matrices=[review_matrix],
                                                        query_matrices=not_review_matrix_list,
                                                        method='euclid_dist_naive',
                                                        sort_descending=sort_descending)

            ranking_matrix[n_review, :] = sort_idx


        if n_review > 0:
            average_rank = np.mean(ranking_matrix, 0)
        else:
            average_rank = ranking_matrix

        sort_idx = np.argsort(average_rank).flatten()   # sort in ascending order

        if query_matrix_indices is not None:
            sorted_indices = query_matrix_indices[sort_idx]
        else:
            sorted_indices = None

    return average_rank, sort_idx, sorted_indices


def compute_matrix_similariry_w_index(matrix_list, template_idx, query_idx, method='euclid_dist_naive'):


    tensor = np.concatenate(matrix_list)



    return similarity_scores.flatten(), sort_idx, sorted_indices


def compute_matrix_similarity_xr(dataarray, template_idx, method='euclid_dist_naive'):


    trial_indices = dataarray['Trial'].values
    template_matrices = dataarray.isel(Trial=template_idx)
    template_matrices_list = [dataarray.isel(Trial=x).values for x in np.arange(len(template_matrices.Trial))]

    query_indices = trial_indices[~np.isin(trial_indices, template_idx)]
    query_matrices = dataarray.isel(Trial=query_indices)
    query_matrice_list = [dataarray.isel(Trial=x).values for x in np.arange(len(query_matrices.Trial))]

    similarity_scores, sort_idx, sorted_indices = compute_matrix_similarity(
        template_matrices=template_matrices_list, query_matrices=query_matrice_list,
        method='euclid_dist_naive', query_matrix_indices=query_indices)


    return similarity_scores, sort_idx, sorted_indices


def compute_ranking_xr(dataarray, template_idx, template_score, method='elucid_dist_naive',
                       query_matrix_indices=None):
    """

    Parameters
    -------------
    dataarray : (xarray datarray object)
        dataarray object containing ALL of the matrices (both the template and the query matrices)
    :param template_idx:
    :param template_score:
    :param method:
    :param query_matrix_indices:
    :return:
    """

    # print(dataarray)

    if query_matrix_indices is None:
        trial_idx = dataarray['Trial'].values
        trial_idx = np.delete(trial_idx, template_idx)
        query_matrix_indices = trial_idx

    dim_names = xr_util.get_ds_dim_names(dataarray)
    if 'Exp' in dim_names:
        dataarray = dataarray.isel(Exp=0)

    matrix_list = [dataarray.isel(Trial=x).values for x in np.arange(len(dataarray.Trial))]
    # print(len(matrix_list))
    # print(np.shape(matrix_list[0]))

    average_rank, sort_idx, sorted_indices = compute_matrix_similarity_w_feedback(matrix_list=matrix_list,
                                         reviewed_idx=template_idx, review_score=template_score,
                                         similarity_calculation_method='euclid_dist_naive',
                                         agg_method='rank', query_matrix_indices=query_matrix_indices)


    return average_rank, sort_idx, sorted_indices
