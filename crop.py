
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

# import math

# iter_x =max_x// step_size
# iter_x

# print("Number of processors: ", mp.cpu_count())

# for i in range(math.ceil(max_x// step_size)):
#     print(i)
iter =range(math.ceil(max_x// step_size))
    
results = []

def howmany_within_range2(i, row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return (i, count)


def collect_result(result):
    global results
    results.append(result)


for i, row in enumerate(data):
    pool.apply_async(crop, args=(iter, row, 4, 8), callback=collect_result)

pool.close()
pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

results.sort(key=lambda x: x[0])
results_final = [r for i, r in results]

print(results_final[:10])
