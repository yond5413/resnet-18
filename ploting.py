import matplotlib.pyplot as plt



workers = [0,1,2,4,8,12,16]
io_times = [5.332523912191391,4.571821928024292, 4.525736033916473,1.242966279387474,
         1.243903398513794,1.229981154203415,1.2434450834989548]
'''
0: 5.332523912191391
1: 4.571821928024292
2: 4.525736033916473
4: 1.242966279387474
8: 1.243903398513794
12: 1.229981154203415
16: 1.2434450834989548
Average Epoch time:63.83890561060107 cuda for 12
verage Epoch time:459.195976296399 cpu
Number of workers: 12 sec
'''
'''
##########
sgd

###########
nesterov

###########
adagrad

###########
adadelta

###########
adam

###########
'''

#SGD Finding Gradients vs parameters
#params: 42, grads: 42
#ADAM
#
print(len(workers))
plt.plot(workers,io_times)
plt.plot(workers,io_times,marker='o', linestyle='', markersize=10, markerfacecolor='red', markeredgecolor='black', label='Highlighted Points')

plt.xlabel("workers")
plt.ylabel("I/O performance")
plt.title("workers vs I/O performance")
plt.show()

#print(workers)