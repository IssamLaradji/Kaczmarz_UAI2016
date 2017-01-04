#%%
import numpy as np
from scipy.sparse import csr_matrix
import kaczmarz as kz
from pretty_plot import pretty_plot
import scipy.sparse as sp
import pandas as pd
from scipy.io import loadmat
import heap_jit as hp
import utilities as ut
import coordinate_descent as cd
import unittest
import adaptive_jit as adept
label_map, title_map, indices_algs, figure_names = ut.get_name_map()

def _train_CD_Regressor(A, b, n_epochs, algorithm, x_axis):
    np.random.seed(0)

    model = cd.CDRegressor(max_epochs=n_epochs, selection_algorithm=algorithm, verbose=x_axis)

    model.fit(A, b)

    return model


def _train_Kaczmarz(A, b, n_epochs, algorithm, x_axis, dataset_name, theta_true):
    np.random.seed(0)

    model = kz.Kaczmarz(n_epochs=n_epochs, algorithm=algorithm, verbose_type=x_axis, dataset_name=dataset_name,
                        x_true=theta_true)

    model.fit(A, b)

    return model

#%% Unit testing
class TestKz(unittest.TestCase):
    
    def test_consistent(self):
         for data_func in [ut.load_very_sparse_data,
                           ut.load_exp_data,
                           ut.load_exp3_data]:
            A, b, n_samples, n_features, dataset_name =  data_func()
            assert ut.isConsistent(A, b)
            
    def test_1_heap_vs_greedy_CD(self):
        # 1. Test coordinate descent such that GS heap = GS greedy, GSL heap = GSL greedy
        for data_func in [ut.load_very_sparse_data, ut.load_exp_data]:
            A, b, n_samples, n_features, dataset_name =  data_func()
            print dataset_name
            A = A.toarray()
            n_epochs = 100
            x_axis = "Iteration"
            model_greedy = _train_CD_Regressor(A, b, n_epochs, "GSL", x_axis)
            model_heap = _train_CD_Regressor(A, b, n_epochs, "GSL_heap", x_axis)
    
            model_greedy_result = pd.DataFrame.from_dict(model_heap.results)
            model_heap_result = pd.DataFrame.from_dict(model_greedy.results)
    
    
            np.testing.assert_array_equal(np.array(model_greedy_result["Selected_Coordinate"]),
                                          np.array(model_heap_result["Selected_Coordinate"]))
    
            print "GSL passed!"
            model_greedy = _train_CD_Regressor(A, b, n_epochs, "GS", x_axis)
            model_heap = _train_CD_Regressor(A, b, n_epochs, "GS_heap", x_axis)
    
            model_greedy_result = pd.DataFrame.from_dict(model_heap.results)
            model_heap_result = pd.DataFrame.from_dict(model_greedy.results)
    
    
            np.testing.assert_array_equal(np.array(model_greedy_result["Selected_Coordinate"]),
                                          np.array(model_heap_result["Selected_Coordinate"]))
    
            print "GS passed!\n==========="
    
    
    def test_2_heap_vs_greedy_Kaszmarz(self):
        # 2. Test Kascmarz such that GS heap = GS greedy, GSL heap = GSL greedy
        for data_func in [ ut.load_very_sparse_data, ut.load_exp_data]:
            A, b, n_samples, n_features, dataset_name =  data_func()
            print dataset_name
            theta_true = ut.get_solution(A.toarray(), b)
    
            n_epochs = 100
    
            results_greedy =ut.fit_kaczmarz_iteration(A, b, n_epochs, "greedy_res", theta_true, dataset_name)
            results_heap  =ut.fit_kaczmarz_iteration(A, b, n_epochs, "heap_rule_residual", theta_true, dataset_name)
    
            
            np.testing.assert_array_equal(np.array(results_greedy["Selected_sample"]),
                                          np.array(results_heap["Selected_sample"]))
            
            #ut.plot_iteration(dataset_name, n_epochs, "Distance", ["greedy_res", "heap_rule_residual"])
    
    
            print "MR passed!"
    
            results_greedy =ut.fit_kaczmarz_iteration(A, b, n_epochs, "greedy_dis", theta_true, dataset_name)
            results_heap  =ut.fit_kaczmarz_iteration(A, b, n_epochs, "heap_rule_distance", theta_true, dataset_name)
    
            
            np.testing.assert_array_equal(np.array(results_greedy["Selected_sample"]),
                                          np.array(results_heap["Selected_sample"]))
            
            #ut.plot_iteration(dataset_name, n_epochs, "Distance", ["greedy_dis", "heap_rule_distance"])
    
    
            print "MD passed!\n==========="
            
    def test_adaptive_jit(self):
        #n_samples = 10
        #elements = np.array([2.9, 4., 6.,10., 1.2, 8., 2., 9., 2.5, 7.])
    
        # Test 1 - propagate up
        elements = np.array([3.,5.,1.])
        tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)
    
        assert np.array_equal(tree_array, np.array([9, 6, 3.,5.,1.]))
    
        elements = np.array([4.,8.,20.,15,-5.,-4])
        tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)
    
        assert np.array_equal(tree_array, np.array([ 38. , 26. , 12.,  35.,  -9.,   4. ,  8. , 20.,  15.,  -5.,  -4.]))
    
    
    
    
        # Test 2 - single propagate up
        new_value = -100
        row_index = 2
    
        adept.single_propagate_up(new_value, row_index, tree_array , n_leaves, n_inner_nodes)
        assert np.array_equal(tree_array, np.array([ -82.,  -94.,   12.,  -85. ,  -9.,    4. ,   8. ,-100. ,  15. ,  -5.,-4.]))
    
    
        new_value = 100
        row_index = 1
        adept.single_propagate_up(new_value, row_index, tree_array , n_leaves, n_inner_nodes)
        assert np.array_equal(tree_array, np.array([ 10.,  -94.,   104.,  -85. ,  -9.,    4. ,  100. ,-100. ,  15. ,  -5.,-4.]))
    
        # Test 3 - batch propagate up
        elements = np.array([4.,8.,20.,15,-5.,-4])
        tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)
    
        new_values = np.array([-100., 100])
        row_indices = np.array([2, 1]).astype(np.int64)
    
        adept.batch_propagate_up(new_values, row_indices, tree_array , n_leaves, n_inner_nodes)
        assert np.array_equal(tree_array, np.array([ 10.,  -94., 104., -85. , -9., 4. , 100. ,-100. , 15. , -5.,-4.]))
    
        n_samples = 10000
        elements_1 = np.random.rand(n_samples)
    
        tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements_1)
    
    
        n_samples = 10000
        elements_2 = np.random.rand(n_samples)
    
        tree_array_2, n_leaves_2, n_inner_nodes_2 = adept.construct_tree(elements_2)
    
        adept.batch_propagate_up(elements_2, np.arange(n_samples).astype(np.int64), tree_array , n_leaves, n_inner_nodes)
    
        assert  abs(np.linalg.norm(tree_array - tree_array_2)) < 1e-5
    
        # Test 4 - test propagate down
        elements = np.array([4.,4.,4.,4.,4.,1,1,1,1.,1])
        tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)
    
        left, right = 0,0
        for i in range(100000):
            k = adept.propagate_down(tree_array, n_leaves, n_inner_nodes)
            assert k >= 0
            assert k < n_leaves
    
            if k < n_leaves / 2:
                left += 1
            else:
                right +=1
        assert abs(float(left)/right - 4) < 0.4
    
        elements = np.array([5.,5.,5.,5.,5.,10,10,10,10.,10])
        tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)
    
        left, right = 0,0
        for i in range(100000):
            k = adept.propagate_down(tree_array, n_leaves, n_inner_nodes)
            assert k >= 0
            assert k < n_leaves
    
            if k < n_leaves / 2:
                left += 1
            else:
                right +=1
    
        assert abs(float(right)/left - 2) < 0.4
        
    def test_major_unit_heap(self):
        n_samples = 100000
        elements = np.random.rand(n_samples)
    
        # Test heap construction
        heap, heap_to_list_index, list_to_heap_index, n_elements = hp.create(elements)
        #hp.batch_insert(heap, heap_to_list_index, list_to_heap_index, elements)
    
        # heap_to_list_index should be consistent with heap
        assert np.array_equal(heap, elements[heap_to_list_index])
    
        # heap elements locations should be consistent with elements
        assert np.array_equal(heap[list_to_heap_index], elements)
    
    
        n_samples_changed = 10000
        index_list = np.arange(n_samples_changed)
    
        elements[index_list] = np.random.rand(n_samples_changed)
        
        # Test batch update
        hp.batch_update(elements, index_list.astype(np.int64), heap, heap_to_list_index, list_to_heap_index, n_samples)
        assert np.array_equal(heap, elements[heap_to_list_index])
        assert heap[0] == np.max(elements)
        assert heap_to_list_index[0] == np.argmax(elements)
        assert np.array_equal(heap[list_to_heap_index], elements)
    
        print "major unit tests passed !"
    
    def test_micro_unit_heap(self):
        n_samples = 10
    
        elements = np.array([2.9, 4., 6.,10., 1.2, 8., 2., 9., 2.5, 7.])
    
        heap, heap_to_list_index, heap_find_ids, n_elements = hp.create(elements)
        #hp.batch_insert(heap, heap_to_list_index, heap_find_ids, elements)
    
        # heap_to_list_index should be consistent with heap
        assert np.array_equal(heap, elements[heap_to_list_index])
    
        # Max and argmax should be correct
        assert np.max(elements) == heap[0]
        assert np.argmax(elements) == heap_to_list_index[0]
    
        # Test batch update
        ##############################
    
        # Test - no sift should happen
        ##############################
        heap_1 = heap.copy()
        heap_to_list_index_1 = heap_to_list_index.copy()
        heap_find_ids_1 = heap_find_ids.copy()
    
        index_list = heap_to_list_index[np.array([8, 5,2,3])]
        new_elements = np.array([5.5, 7.5,9.2,6.5])
    
        hp.batch_update(new_elements, index_list.astype(np.int64), heap_1, heap_to_list_index_1, heap_find_ids_1, n_samples)
    
        assert np.array_equal(heap_1, np.array([ 10.,9.,9.2,6.5,7.,7.5 ,2.,2.9,5.5,1.2]))
    
        # Test - sift up
        ##############################
        heap_1 = heap.copy()
        heap_to_list_index_1 = heap_to_list_index.copy()
        heap_find_ids_1 = heap_find_ids.copy()
    
        index_list = heap_to_list_index_1[np.array([5,9,7,8])]
        new_elements = np.array([ 11,12,7.5,8])
        hp.batch_update(new_elements, index_list.astype(np.int64), heap_1, heap_to_list_index_1, heap_find_ids_1, n_samples)
    
        assert np.array_equal(heap_1, np.array([ 12.,11.,10,8,9,8 ,2.,6,7.5,7]))
    
        # Test - sift down
        ##############################
        heap_1 = heap.copy()
        heap_to_list_index_1 = heap_to_list_index.copy()
        heap_find_ids_1 = heap_find_ids.copy()
    
        for index_, element_ in zip(np.array([1,1,4,0]), np.array([5,1,0.5,3])):
            h_index = heap_to_list_index_1[index_]
            hp.batch_update(np.array([element_]), np.array([h_index]).astype(np.int64), heap_1, heap_to_list_index_1, heap_find_ids_1, n_samples)
    
        assert np.array_equal(heap_1, np.array([ 8.,6.,4,2.9,1.2,3 ,2.,1,2.5,0.5]))


if __name__ == '__main__':
    unittest.main()
