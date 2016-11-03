# Multiple-kernal-scikit-svr
A Kernel Maker Module which can easily be integrated with sklearn's SVR module
A Sample Use Case:
```
x=np.matrix([[1,2],[2,4],[3,6],[4,8],[5,10]])
y=np.array([2,4,6,8,10])
a=multi_kernel_maker(x,y,[lin(),rbf(),sig(),poly(power=2)])
svr=SVR(kernel=a)
c=svr.fit(x,np.squeeze(np.asarray(y)))
x_new=np.matrix([[6,10]])
y_new=svr.predict(x_new)
y_test=svr.predict(x)
print c,y_new,y_test
```
Here `[lin(),rbf(),...]` is a list of kernel functions

`lin(offset=0)` creates a linear kernal function of `x1.x2+offset`
`poly(power=3,offset=0)` creates a polynomial kernal function of `(x1.x2+offset)**power`
`rbf(gamma=1.0)` creates a radial basis kernal function of `e**(||x2-x1||*-gamma)`
`sig(alpha=1.0,offset=0)` creates a sigmoid kernal function of `tanh(alpha*x1.x2+offset)`

You can also add your own kernal functions to the kernal list
