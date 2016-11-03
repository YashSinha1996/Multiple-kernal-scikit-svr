# Multiple-kernal-scikit-svr
A Kernel Maker Module which can easily be integrated with sklearn's SVR module
`x=np.matrix([[1,2],[2,4],[3,6],[4,8],[5,10]])
y=np.array([2,4,6,8,10])
a=multi_kernel_maker(x,y,[lin(),rbf(),sig(),poly(power=2)])
svr=SVR(kernel=a)
c=svr.fit(x,np.squeeze(np.asarray(y)))

# svr predict needs same number of samples as used in traing.. need to fix that
# Fixed it :)
x_new=np.matrix([[6,10]])
y_new=svr.predict(x_new)
y_test=svr.predict(x)
print c,y_new,y_test`
