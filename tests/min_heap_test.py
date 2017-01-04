import numpy as np
import time
import heap_jit as hp
#import heap_ as hp

def major_unit_tests():
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

def micro_unit_tests():
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

    print "micro unit tests passed !"

s = time.time()
major_unit_tests()
micro_unit_tests()
print "Time taken", time.time() - s