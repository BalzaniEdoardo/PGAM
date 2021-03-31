library(reticulate)
library(glmnet)

np <-import("numpy")
data <- np$load("/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/logRegModelX.npy")
Y <- np$load("/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/logRegModelY.npy")
ppc <- data[,1]
delta_mst <- data[,2]
delta_pfc <- data[,3]
b1 <- data[,4]
b2 <- data[,5]
b3 <- data[,6]
b4 <- data[,7]
b5 <- data[,8]
b6 <- data[,9]
d.AD <- data.frame( as.integer(Y),delta_mst, delta_pfc,b1,b2,b3,b4,b5,b6)
glmMod = glm(Y ~ delta_mst + delta_pfc + b1 + b2 + b3 + b4 + b5 + b6, family = binomial())
anova(glmMod)
summary(glmMod)

res = glmnet(data, as.factor(Y),family="binomial",alpha=1,lambda.min.ratio=0.01)

res$beta = as.matrix(res$beta)
plot(res, xvar="lambda")

n = dim(data)[1]
train_rows <- sample(1:n, .90*n)

x.train <- data[train_rows, ]
x.test <- data[-train_rows, ]

y.train <- as.factor(Y)[train_rows]
y.test <- as.factor(Y)[-train_rows]

resTrain = glmnet(x.train,y.train,family="binomial",alpha=1,lambda.min.ratio=0.0000001)
resTrain$beta = as.matrix(resTrain$beta)

muTest = predict(resTrain, x.test)
deviance(resTrain)
dev_function(y.test, muTest[,37], c(rep(1, length(y.test))), 'binomial')
#fit.lasso.cv <- cv.glmnet(x.train, y.train, type.measure="deviance", alpha=1, 
#                          family="binomial")
