#Feature selection process by Lasso
#@author: Li Na == Sun Yat-sen University
#@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn

library(openxlsx)
library(readxl)
library(glmnet)

#Read data
df<-read_excel("fdata.xlsx")
View(df)
X <- df[, c(1: 30)]
y <- data.matrix(df$Drugused)
#Build lasso model
f1=glmnet(X,y,family = "binomial",nlambda = 100,alpha = 1)
print(f1)
windows(width = 16, height = 12)
plot(f1,xvar="lambda",label="TRUE")
#Feature selection using LASSO
lasso_fit <- cv.glmnet(data.matrix(X), y, family="binomial", alpha=1,
                       nfolds = 5, set.seed(1314))
plot(lasso_fit)
#Get the optimal model
coef1<-coef(lasso_fit,s="lambda.min",exact = F)
coef1