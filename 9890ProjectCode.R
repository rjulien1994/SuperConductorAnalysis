library(randomForest)
library(glmnet)
library(ggplot2)
library(gridExtra)

#sets the folder in which the script is as wrking directory
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

par(mfrow=c(1,1)) #output a single graph at a time
par(mar=c(2,2.3,1.6,1.6))  #set up graph spacing

maxN = 21263
r = 100   #number of times we want to repeat each analysis 
          #it is worth noting that randomForest seems to be n(O) complexity

#assuming you have R
superCon.df = read.csv('9890ProjectMod.csv')[1:maxN,]

head(superCon.df)       #We see the first few element of the df
sum(is.na(superCon.df)) #we check for empty records

nObs = dim(superCon.df)[1]
p = dim(superCon.df)[2]-1
ypos = p+1   #if response variable not last change value

X.original = data.matrix(superCon.df[-(ypos)], rownames.force = NA)
y.original = data.matrix(superCon.df[(ypos)], rownames.force = NA)

xSd = apply(X.original,2,'sd')  #computes (x - xBar)^2 /n
xMean = apply(X.original,2,'mean')

#X transformations
#standartize based on formula 6.6 in ISLR (no centering)
varMatrix = diag(p) * 1/xSd  #store it in diag matrix for algebraic formula
X.standart = X.original %*% varMatrix  #divide each x by the attribute sd

#centers and scales
X.scaled = X.original
for (i in c(1:nObs)){
  X.scaled[i,]   =    (X.original[i,] - xMean)/xSd
}

#Y transformations
par(mfrow=c(2,1))

y.logged = log(y.original)
hist(y.original)    #Helps us choose if we use logged or original
hist(y.logged)    

y = y.logged  #select which transformation of data we want to use for analysis
X = X.scaled  #I get a lower error using scaled than standartized

nTrain = floor(0.8*nObs)  #counts the number of obs in train and test 
fold = (1:nTrain)

steps = 1:r
train.Rsqr = list(ridge = steps, lasso = steps, elastic = steps, randForest = steps)
test.Rsqr = list(ridge = steps, lasso = steps, elastic = steps, randForest = steps)

compTime = c(0,0,0,0)  #store time in millis to see computing efficiency

for (s in steps) {

  i.mix = sample(1:nObs)  #we randomize the selected datapoints

  X.train = X.standart[i.mix[fold],] #we store the selected datapoint in their respective group
  y.train = y[i.mix[fold]]
  X.test = X.standart[i.mix[-fold],]
  y.test = y[i.mix[-fold]]

  #Ridge Method
  #if we want to specify our lambda range 
  #lambda.range = c(seq(0.0001,0.005,length=30), seq(0.005,0.1,length=30), seq(0.1,10,length=30))
  startTime = as.numeric(Sys.time())*1000
    
  cv.ridge = cv.glmnet(X.train, y.train, family="gaussian", alpha = 0, nfolds = 10)
  fit.ridge = glmnet(X.train, y.train, alpha = 0,family="gaussian", lambda = cv.ridge$lambda.min)

  y.train.hat = predict(fit.ridge, newx = X.train, type = "response") 
  y.test.hat = predict(fit.ridge, newx = X.test, type = "response")
  train.Rsqr$ridge[s] = 1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  test.Rsqr$ridge[s] = 1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  stopTime = as.numeric(Sys.time())*1000
  compTime[1] = compTime[1] + stopTime - startTime

  #Lasso Method
  startTime = as.numeric(Sys.time())*1000

  cv.lasso = cv.glmnet(X.train, y.train, alpha = 1,family="gaussian", nfolds = 10)
  fit.lasso = glmnet(X.train, y.train, alpha = 1, lambda = cv.lasso$lambda.min)
  
  y.train.hat = predict(fit.lasso, newx = X.train, type = "response")
  y.test.hat = predict(fit.lasso, newx = X.test, type = "response")
  train.Rsqr$lasso[s] = 1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  test.Rsqr$lasso[s] = 1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  stopTime = as.numeric(Sys.time())*1000
  compTime[2] = compTime[2] + stopTime - startTime

  #elastic Method
  startTime = as.numeric(Sys.time())*1000

  cv.elastic = cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  fit.elastic = glmnet(X.train, y.train, alpha = 0.5, lambda = cv.elastic$lambda.min)

  y.train.hat = predict(fit.elastic, newx = X.train, type = "response")
  y.test.hat = predict(fit.elastic, newx = X.test, type = "response")
  train.Rsqr$elastic[s] = 1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  test.Rsqr$elastic[s] = 1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  stopTime = as.numeric(Sys.time())*1000
  compTime[3] = compTime[3] + stopTime - startTime

  #random forest
  startTime = as.numeric(Sys.time())*1000

  rf = randomForest(X.train, y.train, ntree = 51, importance = TRUE)
  #I chose 51 trees for efficiency and error doesn't seem to change much afterwards
  
  y.train.hat = predict(rf, X.train)
  y.test.hat = predict(rf, X.test)
  train.Rsqr$randForest[s] = 1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  test.Rsqr$randForest[s] = 1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  #if we use factors, rf returns a confusion matrix
  
  stopTime = as.numeric(Sys.time())*1000
  compTime[4] = compTime[4] + stopTime - startTime
  
  print(s)
}


print(compTime/r) #avg time in millisecond of analysis

#plot for r-squared
par(mfrow=c(1,1))
par(mar=c(3,6,2,1))
boxplot(train.Rsqr$ridge, test.Rsqr$ridge, 
        train.Rsqr$elastic, test.Rsqr$elastic, 
        train.Rsqr$lasso, test.Rsqr$lasso,
        train.Rsqr$randForest, test.Rsqr$randForest,
        main="R-squared boxPlot", 
        at = c(1,2,4,5,7,8,10,11),
        names=c("train Ridge","test Ridge", "train Elastic","test Elastic","train Lasso","test Lasso", "train rf","test rf"),
        las = 2,
        col = c("orange","red"),
        horizontal=TRUE)

#plot for lambda tuning in CV
par(mfrow=c(2,2))
par(mar=c(2,2,1,1))
plot(log(cv.ridge$lambda),cv.ridge$cvm,pch=19,col="red", main="Ridge")
plot(log(cv.elastic$lambda),cv.elastic$cvm,pch=19,col="grey", main="Elastic")
plot(log(cv.lasso$lambda),cv.lasso$cvm,pch=19,col="blue", main="Lasso")

plot(log(cv.ridge$lambda),cv.ridge$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cv.ridge$name,xlim = c(-8,6), main="Comparaison")
points(log(cv.elastic$lambda),cv.elastic$cvm,pch=19,col="grey")
points(log(cv.lasso$lambda),cv.lasso$cvm,pch=19,col="blue")

#plots for residuals
train.Res = list(ridge = as.vector(y.train - predict(fit.ridge, newx = X.train, type = "response")), 
                 lasso = as.vector(y.train - predict(fit.lasso, newx = X.train, type = "response")), 
                 elastic = as.vector(y.train - predict(fit.elastic, newx = X.train, type = "response")), 
                 randForest = as.vector(y.train - predict(rf, X.train, type = "response")))
test.Res = list(ridge = as.vector(y.test - predict(fit.ridge, newx = X.test, type = "response")), 
                 lasso = as.vector(y.test - predict(fit.lasso, newx = X.test, type = "response")), 
                 elastic = as.vector(y.test - predict(fit.elastic, newx = X.test, type = "response")), 
                 randForest = as.vector(y.test - predict(rf, X.test)))

par(mfrow=c(1,1))
par(mar=c(3,6,2,1))
boxplot(train.Res$ridge, test.Res$ridge, 
        train.Res$elastic, test.Res$elastic, 
        train.Res$lasso, test.Res$lasso,
        train.Res$randForest, test.Res$randForest,
        main="BoxPlot for residuals", 
        at = c(1,2,4,5,7,8,10,11),
        names=c("train Ridge","test Ridge", "train Elastic","test Elastic","train Lasso","test Lasso", "train rf","test rf"),
        las = 2,
        col = c("orange","red"),
        horizontal=TRUE)

#bootstrap for coefficient estimation
coeff.bs = list(ridge = matrix(0, nrow = p, ncol = r), 
                lasso = matrix(0, nrow = p, ncol = r), 
                elastic = matrix(0, nrow = p, ncol = r), 
                randForest = matrix(0, nrow = p, ncol = r)) 


for (s in 1:r){
  bs_indexes       =     sample(nObs, replace=T)  #allows duplicate entries
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  #ridge
  cv.ridge = cv.glmnet(X.bs, y.bs, family="gaussian", alpha = 0, nfolds = 10)
  fit.ridge = glmnet(X.bs, y.bs, alpha = 0,family="gaussian", lambda = cv.ridge$lambda.min)
  coeff.bs$ridge[,s] = as.vector(fit.ridge$beta)
  
  #elastic-net
  cv.elastic = cv.glmnet(X.bs, y.bs, family="gaussian", alpha = 0.5, nfolds = 10)
  fit.elastic = glmnet(X.bs, y.bs, alpha = 0.5,family="gaussian", lambda = cv.elastic$lambda.min)
  coeff.bs$elastic[,s] = as.vector(fit.elastic$beta)
  
  #lasso
  cv.lasso = cv.glmnet(X.bs, y.bs, family="gaussian", alpha = 1, nfolds = 10)
  fit.lasso = glmnet(X.bs, y.bs, alpha = 1,family="gaussian", lambda = cv.lasso$lambda.min)
  coeff.bs$lasso[,s] = as.vector(fit.lasso$beta)
  
  #random forest
  rf = randomForest(X.bs, y.bs, ntree = 51, importance = TRUE)
  coeff.bs$randForest[,s] = as.vector(rf$importance[,1])
  
  print(s)
}

bootstrap = list(ridge = data.frame(var=1:p,mean=apply(coeff.bs$ridge, 1, "mean"),sd=apply(coeff.bs$ridge, 1, "sd")),
                 elastic = data.frame(var=1:p,mean=apply(coeff.bs$elastic, 1, "mean"),sd=apply(coeff.bs$elastic, 1, "sd")),
                 lasso = data.frame(var=1:p,mean=apply(coeff.bs$lasso, 1, "mean"),sd=apply(coeff.bs$lasso, 1, "sd")),
                 randForest = data.frame(var=1:p,mean=apply(coeff.bs$randForest, 1, "mean"),sd=apply(coeff.bs$randForest, 1, "sd")))

plot1 = ggplot(bootstrap$ridge) +
  geom_bar( aes(x=var, y=mean), stat="identity", fill="skyblue", alpha=0.7) +
  geom_errorbar( aes(x=var, ymin=mean-sd, ymax=mean+sd), width=0.4, colour="orange", alpha=0.9, size=1.3) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank())

plot2 = ggplot(bootstrap$elastic) +
  geom_bar( aes(x=var, y=mean), stat="identity", fill="skyblue", alpha=0.7) +
  geom_errorbar( aes(x=var, ymin=mean-sd, ymax=mean+sd), width=0.4, colour="orange", alpha=0.9, size=1.3) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank())

plot3 = ggplot(bootstrap$lasso) +
  geom_bar( aes(x=var, y=mean), stat="identity", fill="skyblue", alpha=0.7) +
  geom_errorbar( aes(x=var, ymin=mean-sd, ymax=mean+sd), width=0.4, colour="orange", alpha=0.9, size=1.3) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank())

plot4 = ggplot(bootstrap$randForest) +
  geom_bar( aes(x=var, y=mean), stat="identity", fill="skyblue", alpha=0.7) +
  geom_errorbar( aes(x=var, ymin=mean-sd, ymax=mean+sd), width=0.4, colour="orange", alpha=0.9, size=1.3) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank())

grid.arrange(plot1, plot2, plot3, plot4, nrow=4)






